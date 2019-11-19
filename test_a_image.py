import torch
import cv2
import os
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# 重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
from PIL import Image

import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
classes = ('cat', 'dog')

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('/root/data/Tom/dogvscatv01/model/model.pkl')  # 加载模型
    if torch.cuda.is_available():
        print('yes')
        device = torch.device("cuda")
        model = model.to(device)
        # model.eval()  # 把模型转为test模式

    print(model)
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

    img = cv2.imread("/root/data/Share data/dogs-vs-cats/test1/10056.jpg")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    output = model(img)
    prob = F.softmax(output, dim=1)  # prob是2个分类的概率
    print(prob)
    value, predicted = torch.max(output.data, 1)
    print(predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print(pred_class)
