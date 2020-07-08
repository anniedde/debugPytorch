#################

import os, sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config # pylint: disable=no-name-in-module
from test import test_model
import utils
from myclasses import CIFAR10Noise

from os.path import join as pathjoin
import shutil, argparse, datetime, json
from IPython import embed
from pynvml import *
from random import randrange
from copy import deepcopy
from collections import OrderedDict


def imshow(img):
    img = img / 5 + 0.48     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

####################

conf = Config(run_number=1)
net, device, handle = utils.prepare_net(conf.net(), conf.use_gpu)
model = (torch.load('/usr/xtmp/CSPlus/VOLDNN/Shared/train_log/cifar10_landscape_exp/resnet_multi_local_min/vgg19_1dplots/run_12/models/final.pth', map_location=torch.device('cpu')))

# if running on CPU
if (device == 'cpu'):
    new_state_dict = OrderedDict()
    for k, v in model.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    net.load_state_dict(new_state_dict) # load model

    # turn off tracking running means in batch norm layers
    for child in net.children():
        if type(child) is (torch.nn.modules.container.Sequential):
            for layer in child:
                if isinstance(layer, nn.BatchNorm2d):
                    layer.track_running_stats = False
                    layer.momentum = 0
# if running on GPU
else:
    net.load_state_dict(model)# load model

    # turn off tracking running means in batch norm layer
    for child in net.module.children():
        if type(child) is (torch.nn.modules.container.Sequential):
            for layer in child:
                if isinstance(layer, nn.BatchNorm2d):
                    layer.track_running_stats = False
                    layer.momentum = 0

train_set = conf.dataset(train=True, transform=conf.train_transform)
train_set2 = CIFAR10Noise()

train_loader = DataLoader(train_set, batch_size=1500, shuffle=False, num_workers=conf.test_provider_count)
train_loader2 = DataLoader(train_set2, batch_size=1500, shuffle=False, num_workers=conf.test_provider_count)

# turn model to evaluation mode
net.eval()

print('check track_running_stats before forward passes:')
for child in net.children():
    if type(child) is (torch.nn.modules.container.Sequential):
        for layer in child:
            if isinstance(layer, nn.BatchNorm2d):
                print(layer.track_running_stats)
                print(layer.momentum)

print('check running weights before forward passes:')
for child in net.children():
    if type(child) is (torch.nn.modules.container.Sequential):
        print(child[1].weight)

# find matching pictures in each dataset
with torch.no_grad():
    
    # iterate through train set 1
    for data in train_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        # iterate through images in train set 1
        for i, img in enumerate(images):

            # for each image, iterate through train set 2
            for data2 in train_loader2:
                images2, labels2 = data2
                outputs2 = net(images2)
                _, predicted2 = torch.max(outputs2.data, 1)

                # iterate through images in train set 2
                for j, img2 in enumerate(images2):

                    # if image in train set 2 matches image from train set 1,
                    # print image and outputs
                    if torch.equal(img, img2):
                        print('image from original train set:')
                        imshow(images[i])
                        print('image from modified train set:')
                        imshow(images2[j])
                        print('prediction from original train set:')
                        print(classes[predicted[i].item()])
                        print('prediction from modified train set:')
                        print(classes[predicted2[j].item()])
                        print('output from original train set:')
                        print(outputs[i])
                        print('output from modified train set:')
                        print(outputs[j])
                        print('ground truth label from original train set:')
                        print(classes[labels[i].item()])
                        print('ground truth label from modified train set:')
                        print(classes[labels2[j].item()])

                        # check to see if batch norm states have changed
                        print('track_running_stats after forward pass:')
                        for child in net.children():
                            if type(child) is (torch.nn.modules.container.Sequential):
                                for layer in child:
                                    if isinstance(layer, nn.BatchNorm2d):
                                        print(layer.track_running_stats)
                                        print(layer.momentum)

                        # check that running means have not changed
                        print('running weights after forward pass:')
                        for child in net.children():
                            if type(child) is (torch.nn.modules.container.Sequential):
                                print(child[1].weight)
