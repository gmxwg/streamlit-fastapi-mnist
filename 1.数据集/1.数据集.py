import os

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

#2. 下载训练集和测试集合
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform,
)

print(f"训练集长度: {len(train_dataset)}")
print(f"测试集长度: {len(test_dataset)}")

img,label = train_dataset[0]

print(train_dataset[0])

plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"digit: {label}")
plt.show()
