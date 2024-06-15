# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
from torch.optim import SGD, Optimizer, Adam
from train_utils import AverageMeter, accuracy
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--arch', type=str, choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')



args = parser.parse_args()

if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print(os.environ.get('CUDA_VISIBLE_DEVICES'))

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = args.epochs
batch_size = 64
learning_rate = args.lr
# image_shape = (28, 28, 1)
# n_input = np.prod(image_shape)
# n_measurement = 100
# d = 784

train_dataset = get_dataset("custom", 'train')
test_dataset = get_dataset("custom", 'test')
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = get_architecture(args.arch, args.dataset).cuda()
criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean').cuda()

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                             #weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             #weight_decay=1e-5)

if args.optimizer == 'Adam':
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
elif args.optimizer == 'SGD':
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=80, gamma=0.1)

top1 = AverageMeter()
top5 = AverageMeter()
losses = AverageMeter()

avg_loss = 0
for epoch in tqdm(range(num_epochs)):
    for data in dataloader:
        img_original, label = data
        img_original = img_original.cuda()
        label = label.cuda()
        # ===================forward=====================
        output = model(img_original).cuda()
        loss = criterion(output, label).cuda()
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ===================log========================
    avg_loss = loss.data
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, avg_loss))

    if epoch % 1 == 0:
        avg_loss = 0
        num = 0
        for data in testloader:
            img_original, label = data
            img_original = img_original.cuda()
            label = label.cuda()

            # ===================Forward=====================
            output = model(img_original)
            # print(output)
            loss = criterion(output, label)

            loss_mean = loss.mean()
            acc1= accuracy(output, label)
            losses.update(loss_mean.item(), img_original.size(0))
            top1.update(acc1[0].item(), img_original.size(0))

        print('Testing Evaluation: loss:{:.4f}, Testing Accuracy:{:.4f}'
              .format(losses.avg, top1.avg))

torch.save({
            'epoch': epoch + 1,
            'arch': 'VGG',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'best.pth.tar'))
