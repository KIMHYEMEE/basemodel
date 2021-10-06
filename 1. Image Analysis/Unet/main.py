# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 17:58:11 2021

@author: USER

Ref.1: https://github.com/milesial/Pytorch-UNet/blob/master/une
Ref.2: https://wjddyd66.github.io/pytorch/Pytorch-Unet/
"""

import torch
import torch.nn as nn
import modeling
from torchvision import transforms

from os import listdir
from os.path import isfile, join
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

img_size = 256
n_epoch = 3000
lr = 0.001
batch_size = 30
n_epoch = 100

data_dir = 'C:/BaseData/Project/5. DBR/202109_data/original/OK'

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

loader = transforms.Compose([
    transforms.ToTensor()
    ])

def img_loader(img_name):
    img = Image.open(img_name)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)


data = []

for i in range(len(files)):
    img = img_loader(data_dir +'/'+ files[i])
    data.append(img)


data_batch = []

for i in range(0, len(data)-batch_size, batch_size):
    data_batch.append(torch.cat([data[k] for k in range(i,i+batch_size)]))

unet = modeling.UnetGenerator().to(device)
print(unet)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(unet.parameters(), lr=lr, momentum=0.99)

for i in range(n_epoch):
    for j in range(len(data_batch)):
        x = data_batch[j]
        optimizer.zero_grad()
        output = unet.forward(x)
        loss = loss(output,x)
        loss.backward()

        optimizer.step()
    
    if i % 10 == 0 :
        print(loss.item())