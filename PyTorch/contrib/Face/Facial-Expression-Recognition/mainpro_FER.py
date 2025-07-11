from __future__ import print_function
from torch_sdaa.utils import cuda_migrate
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
import time


from torch.sdaa import amp
scaler = torch.sdaa.amp.GradScaler()

# === Tecorign 日志系统 ===
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
json_logger = Logger([
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
])
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

# === 参数解析 ===
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
best_PublicTest_acc = 0
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0
best_PrivateTest_acc_epoch = 0
start_epoch = 0

learning_rate_decay_start = 80
learning_rate_decay_every = 5
learning_rate_decay_rate = 0.9
cut_size = 44
total_epoch = 250
path = os.path.join(opt.dataset + '_' + opt.model)

# === 数据预处理 ===
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
PublicTestset = FER2013(split='PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)

# === 模型 ===
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model == 'Resnet18':
    net = ResNet18()

if opt.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PublicTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')

net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# === 训练函数（最多100步） ===
def train(epoch, global_step, max_steps):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if global_step >= max_steps:
            break

        inputs = inputs.to(device).to(memory_format=torch.channels_last)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.sdaa.amp.autocast():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        utils.clip_gradient(optimizer, 0.1)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(batch_idx, len(trainloader),
                           'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                           (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        elapsed_time = time.time() - start_time
        ips = (batch_idx + 1) * opt.bs / elapsed_time if elapsed_time > 0 else 0
        json_logger.log(
            step=(epoch, batch_idx),
            data={
                "rank": int(os.environ.get("LOCAL_RANK", 0)),
                "train.loss": loss.item(),
                "train.ips": ips
            },
            verbosity=Verbosity.DEFAULT
        )

        global_step += 1

    return global_step

# === PublicTest函数 ===
def PublicTest(epoch):
    global best_PublicTest_acc, best_PublicTest_acc_epoch
    net.eval()
    correct, total, PublicTest_loss = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w).to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    if acc > best_PublicTest_acc:
        print('Saving.. (PublicTest)')
        print("best_PublicTest_acc: %0.3f" % acc)
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_acc = acc
        best_PublicTest_acc_epoch = epoch

# === PrivateTest函数 ===
def PrivateTest(epoch):
    global best_PrivateTest_acc, best_PrivateTest_acc_epoch
    net.eval()
    correct, total, PrivateTest_loss = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w).to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    if acc > best_PrivateTest_acc:
        print('Saving.. (PrivateTest)')
        print("best_PrivateTest_acc: %0.3f" % acc)
        state = {
            'net': net.state_dict(),
            'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': acc,
            'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))
        best_PrivateTest_acc = acc
        best_PrivateTest_acc_epoch = epoch

# === 主训练流程（最多100步） ===
max_steps = 100
global_step = 0
for epoch in range(start_epoch, total_epoch):
    if global_step >= max_steps:
        break
    global_step = train(epoch, global_step, max_steps)
    PublicTest(epoch)
    PrivateTest(epoch)

print("==> Training done.")
print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
