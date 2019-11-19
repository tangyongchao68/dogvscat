# -*- coding: utf-8 -*-
import os
import shutil  # 用来移动图片的库，直接移走


def preprocess_data():
    data_file = os.listdir("/root/data/Share data/dogs-vs-cats/train")  # 读取所有图片的名字
    # 将图片名为cat和dog的图片分别取出来，存为两个list
    # lambda is a function without name
    # 打开文件
    path = "/root/data/Share data/dogs-vs-cats/train"
    dirs = os.listdir(path)
    print(dirs)

    # 输出所有文件和文件夹
    for file in dirs:
        print(file)

    cat_file = list(filter(lambda x: x[:3] == 'cat', data_file))
    dog_file = list(filter(lambda x: x[:3] == 'dog', data_file))

    data_root = '/root/data/Share data/dogs-vs-cats/train/'
    train_root = '/root/data/Tom/dogvscatv01/data/train'
    val_root = '/root/data/Tom/dogvscatv01/data/val'

    for i in range(len(cat_file)):
        print(i)
        pic_path = data_root + cat_file[i]
        if i < len(cat_file) * 0.9:
            obj_path = train_root + '/cat/' + cat_file[i]
        else:
            obj_path = val_root + '/cat/' + cat_file[i]
        shutil.move(pic_path, obj_path)

    for j in range(len(dog_file)):
        print(j)  # 查看进度
        pic_path = data_root + dog_file[j]
        if j < len(dog_file) * 0.9:
            obj_path = train_root + '/dog/' + dog_file[j]
        else:
            obj_path = val_root + '/dog/' + dog_file[j]
        shutil.move(pic_path, obj_path)


# 程序运行接口，调用函数
if __name__ == '__main__':
    preprocess_data()
    data_file_cat = os.listdir('/root/data/Tom/dogvscatv01/data/train/cat')  # 读取所有图片的名字
    print(len(data_file_cat))  # 查看数据大小
