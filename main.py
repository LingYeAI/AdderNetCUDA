#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os

from torch.autograd.grad_mode import F
# from resnet20 import resnet20
from vgg import vgg16
from densenet import densenet_cifar
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
import time
import wandb
import numpy as np

wandb.init(project="DenseNet")

parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./data/')
parser.add_argument('--output_dir', type=str, default='./models/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0
train_loss = 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10(args.data,
                   transform=transform_train,
                   download=True)
data_test = CIFAR10(args.data,
                  train=False,
                  transform=transform_test,
                  download=True)

data_train_loader = DataLoader(data_train, batch_size=216, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)

net = densenet_cifar().cuda()
# net = torch.load("./models/addernet.pt")
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
wandb.watch(net)
def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/400*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win,train_loss
    # train_loss = 0
    net.train()
    loss_list, batch_list = [], []
    old_t = time.time()
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        if images.size(0) != 216:
            break
 
        optimizer.zero_grad()

        output = net(images)
 
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        
        new_t = time.time()
        # if i % 20 == 1:
        # train_loss += loss.data.item()
        print('Train - Epoch %d, Batch: %d, Loss: %f, Time %3f' % (epoch, i, loss.data.item(), new_t - old_t))
        # wandb.log({"Epoch":epoch,"Batch":i,"Loss":loss.data.item(),"Time":new_t - old_t})
        # print("batch time:{}s".format(new_t - old_t))
        old_t = new_t
 
        loss.backward()
        optimizer.step()
    train_loss = np.mean(loss_list)
 
 
def test(e):
    global acc, acc_best,train_loss
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    wandb.log({"Epoch":e, "Test Loss":avg_loss.data.item(),"Train loss":train_loss,"Accuracy":acc})
 
 
def train_and_test(epoch):
    train(epoch)
    test(epoch)
 
 
def main():
    epoch = 401
    # net = torch.load("./models/addernet.pt")
    # net.load_state_dict(torch.load("./models/addernet.pt"))
    for e in range(1, epoch):
        train_and_test(e)
        if e%20 == 0:
            torch.save(net.state_dict(),args.output_dir + 'densenet' + str(acc_best))
 
 
if __name__ == '__main__':
    main()
