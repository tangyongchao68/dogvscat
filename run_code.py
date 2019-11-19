# -*- coding: utf-8 -*-
"""
Created on Thursday November 7 9:00:00 2019

@author: Tom
"""
import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# path = "data"
path = "/root/data/Tom/dogvscatv01/data"
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# Given the mean: (R, G, b) variance: (R, G, b), the tensor will be regularized.
# Namely: normalized_image = (image - mean) / STD.

# The imagefolder function automatically divides cat and dog folders into 0 and 1 categories

data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),
                                      transform=transform)
              for x in ["train", "val"]}
# batch_size refers to how many data in the original data are trained at the same time
data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                    batch_size=4,
                                                    shuffle=True)
                     for x in ["train", "val"]}

# 检查电脑GPU资源
use_gpu = torch.cuda.is_available()
print(use_gpu)  # 查看用没用GPU，用了打印True，没用打印False

# 选择模型
model = models.vgg19(pretrained=True)  # 我们选择预训练好的模型vgg19
print(model)  # 查看模型结构

for parma in model.parameters():
    parma.requires_grad = False  # 不进行梯度更新

# 改变模型的全连接层，因为原模型是输出1000个类，本项目只需要输出2类# 25088
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

if use_gpu:
    model = model.cuda()

# 定义代价函数
cost = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.classifier.parameters())

# 再次查看模型结构
print(model)

# Start training,n_epochs is the number of workouts.
n_epochs = 1
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            y_pred = model(X)
            # torch.max(,1) return the largest number in a row.
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            # running_loss += loss.data[0]
            running_correct += torch.sum(pred == y.data)
            if batch % 5 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct / len(data_image[param])

        print("{} Loss:{:.4f}, Correct:{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))
    # 2 ways to save the net
    torch.save(model, './model/model.pkl')  # save entire net
    torch.save(model.state_dict(), './model/model_params.pkl')  # save only the parameters

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))
