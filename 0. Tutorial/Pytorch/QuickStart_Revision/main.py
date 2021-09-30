# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:27:32 2021

@author: USER

Ref: https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html

Torchvision dataset: https://pytorch.org/vision/stable/datasets.html
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# 1. Load data set
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
    )

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
    )

# 2. Data formatting
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of Y: ", y.shape, y.dtype)
    break


# 3. Modeling
import modeling

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = modeling.NeuralNetwork().to(device)
print(model)

# 4. Training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

from model_fn import train, test
    
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")


# 5. Save model
model_path = 'C:/GIT/basemodel/0. Tutorial/Pytorch'
torch.save(model.state_dict(), f"{model_path}/model.pth")
print("Saved PyTorch Model State to model.pth")


# 6. Load model
model = modeling.NeuralNetwork()
model.load_state_dict(torch.load(f"{model_path}/model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')