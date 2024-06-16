import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD, Adam
from utils import *
from tqdm import tqdm
from config import *
from dataset import BrainTumorDataset
from architech import *
import logging

# Uncomment and use argparse if needed
# import argparse
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                     help='initial learning rate', dest='lr')
# parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
# parser.add_argument('--epochs', default=10, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--arch', type=str, choices=CLASSIFIERS_ARCHITECTURES)
# parser.add_argument('--dataset', type=str, choices=DATASETS)
# parser.add_argument('--optimizer', default='Adam', type=str,
#                     help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
# parser.add_argument('--gpu', default=None, type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')

# args = parser.parse_args()

# Use the config or argparse arguments to set these variables
# lr = args.lr
# epochs = args.epochs
# device = args.gpu if args.gpu else 'cuda:0'
# OUTDIR_TRAIN = args.outdir

os.environ['CUDA_VISIBLE_DEVICES'] = device


test_dataset = BrainTumorDataset(TESTING_ANNOTATION_PATH, TESTING_NEW_DIR)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = Classification().train().cuda()
# model = VGG("VGG19").eval().cuda()
model = ResNet50().eval().cuda()
model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean').cuda()


acc_meter = AverageMeter()
losses_meter = AverageMeter()
acc_avg = 0

for data in tqdm(test_loader):
    imgs, labels = data
    imgs = imgs.cuda()
    labels = labels.cuda()
    output = model(imgs)
    loss = criterion(output, labels)
    acc = accuracy(output, labels)
    acc_avg += acc[0].item()
    # if acc != 1:
    #     print(labels)
    
    losses_meter.update(loss.item(), imgs.shape[0])
    acc_meter.update(acc[0].item(), imgs.shape[0])

print(f"Loss: {losses_meter.avg:.4f} Acc: {acc_meter.avg:.4f}")
print(acc_avg / len(test_dataset))