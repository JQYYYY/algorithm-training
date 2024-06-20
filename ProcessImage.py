import os
import random

import cv2
import numpy as np

train = []
valid = []
test = []
image_size = 224

base_path = 'C:\\Users\\86133\\PycharmProjects\\new_machine learning\\final'
num_classes = 2
# 遍历读取数据
def get_data_names(datatype):
    image_classes = os.listdir(base_path + '\\' + datatype)
    image_names = []
    for i in image_classes:
        names = os.listdir(base_path + '\\' + datatype + '\\' + i)
        image_names.extend(names)
    random.seed(42)
    random.shuffle(image_names)
    return image_names


def get_train_valid_test():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    y_test = []
    image_names = get_data_names('train')
    # print(image_names)
    for name in image_names:
        if 'airplane' in name:
            path = base_path + '\\' + 'train' + '\\00_airplane\\' + name
        else:
            path = base_path + '\\' + 'train' + '\\01_forest\\' + name
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))

        x_train.append(image)
        if 'airplane' in name:
            label = 0
        else:
            label = 1
        y_train.append(label)
    x_train = np.array(x_train, dtype='float32') / 255.0
    y_train = np.array(y_train)

    image_names = get_data_names('valid')
    for name in image_names:
        if 'airplane' in name:
            path = base_path + '\\' + 'valid' + '\\00_airplane\\' + name
        else:
            path = base_path + '\\' + 'valid' + '\\01_forest\\' + name
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        x_valid.append(image)
        if 'airplane' in name:
            label = 0
        else:
            label = 1
        y_valid.append(label)
    x_valid = np.array(x_valid, dtype='float32') / 255.0
    y_valid = np.array(y_valid)

    image_names = get_data_names('test')
    for name in image_names:
        if 'airplane' in name:
            path = base_path + '\\' + 'test' + '\\00_airplane\\' + name
        else:
            path = base_path + '\\' + 'test' + '\\01_forest\\' + name
        image = cv2.imread(path)
        image = cv2.resize(image, (image_size, image_size))
        x_test.append(image)
        if 'airplane' in name:
            label = 0
        else:
            label = 1
        y_test.append(label)
    x_test = np.array(x_test, dtype='float32') / 255.0
    y_test = np.array(y_test)

    return x_train, x_valid, x_test, y_train, y_valid, y_test