import os

import cv2
import keras
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import numpy as np
import pathlib

# path = 'C:\\Users\\86133\\PycharmProjects\\machine learning exam\\airplane\\'
# save_path = 'C:\\Users\\86133\\PycharmProjects\\machine learning exam\\img_data\\airplane'
# tif_list = [x for x in os.listdir(path) if x.endswith('.tif')]
# for i in tif_list:
#     img = cv2.imread(path + i)
#     cv2.imwrite(save_path + '\\' + i.split('.')[0] + '.jpg', img)
#
# path = 'C:\\Users\\86133\\PycharmProjects\\machine learning exam\\forest\\'
# save_path = 'C:\\Users\\86133\\PycharmProjects\\machine learning exam\\img_data\\forest'
# tif_list = [x for x in os.listdir(path) if x.endswith('.tif')]
# for i in tif_list:
#     img = cv2.imread(path + i)
#     cv2.imwrite(save_path + '\\' + i.split('.')[0] + '.jpg', img)


data_dir = "img_data"
data_dir = pathlib.Path(data_dir)

batch_size = 10
img_height = 256
img_weight = 256

train_ds = keras.utils.image_da(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_weight),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    subset='validation',
    seed=123,
    image_size=(img_height, img_weight),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)
