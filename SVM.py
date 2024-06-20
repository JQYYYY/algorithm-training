import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC


def load_image():
    path = 'C:\\Users\\86133\\PycharmProjects\\work\\final1'
    xg = '\\'

    # 读取训练集图片名称
    train_names_list = []
    test_names_list = []
    classes = os.listdir(path + xg + 'train')
    for cl in classes:
        train_names_list.extend(os.listdir(path + xg + 'train' + xg + cl))
        test_names_list.extend(os.listdir(path + xg + 'test' + xg + cl))
    random.seed(42)
    random.shuffle(train_names_list)
    random.shuffle(test_names_list)

    # 将图片转化为灰度图，并将灰度值保存在数组中
    train_data = np.zeros((160, 256, 256, 3))
    test_data = np.zeros((40, 256, 256, 3))
    train_label = []
    test_label = []
    train_index = test_index = 0

    for name in train_names_list:
        if name in os.listdir(path + xg + 'train' + xg + classes[0]):
            image = cv2.imread(path + xg + 'train' + xg + classes[0] + xg + name)
            train_label.append(0)
        else:
            image = cv2.imread(path + xg + 'train' + xg + classes[1] + xg + name)
            train_label.append(1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))
        train_data[train_index, :, :] = image
        train_index += 1

    for name in test_names_list:
        if name in os.listdir(path + xg + 'test' + xg + classes[0]):
            image = cv2.imread(path + xg + 'test' + xg + classes[0] + xg + name)
            test_label.append(0)
        else:
            image = cv2.imread(path + xg + 'test' + xg + classes[1] + xg + name)
            test_label.append(1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))
        test_data[test_index, :, :] = image
        test_index += 1
    # print(train_label)
    train_data = train_data.astype(np.uint8)
    test_data = test_data.astype(np.uint8)
    return train_data, train_label, test_data, test_label


def figure_in_sample(data, s):
    data = np.array(data)
    c = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.figure(figsize=(10, 10))
    for i in range(10):
        start = i * 30
        end = start + 30
        gam = [x for x in range(10, 301, 10)]
        plt.plot(gam, data[start:end], label='C=' + str(c[i]))
    plt.legend()
    plt.xlabel('gamma')
    plt.ylabel(s + ' accuracy')
    plt.title('Accuracy Rate Curve')
    plt.savefig(s + '0.jpg')
    # plt.show()


# 将三通道图片转化为概率直方图
train_hist = np.zeros((160, 256))
test_hist = np.zeros((40, 256))
train_data, train_label, test_data, test_label = load_image()
for i in np.arange(160):
    train_hist[i], _ = np.histogram(train_data[i], density=True, bins=256)
for i in np.arange(40):
    test_hist[i], _ = np.histogram(test_data[i], density=True, bins=256)
# print(train_hist)

bins = np.array([i for i in range(256)])
plt.bar(bins, train_hist[0], width=1, label='Airplane')
plt.bar(bins, train_hist[1], width=1, label='Forest')
plt.xlabel('Bins')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.legend()
plt.savefig('hist1.jpg')

# 调参
# C = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# gamma = [x for x in range(10, 301, 10)]
# train_score = []
# test_score = []
# for c in C:
#     for ga in gamma:
#         estimator = SVC(kernel='rbf', C=c, gamma=ga)
#         estimator.fit(train_hist, train_label)
#         train_score.append(estimator.score(train_hist, train_label))
#         test_score.append(estimator.score(test_hist, test_label))
# figure_in_sample(train_score, 'train')

# 预测测试集
estimator = SVC(kernel='rbf', C=10, gamma=100)
estimator.fit(train_hist, train_label)
test_pred = estimator.predict(test_hist)
score = estimator.score(test_hist, test_label)
print(score)

# 混淆矩阵
# cm = confusion_matrix(test_label, test_pred)
# cm_display = ConfusionMatrixDisplay(cm).plot()
# plt.show()

# ROC曲线
# RocCurveDisplay.from_predictions(
#     test_label,
#     test_pred,
#     plot_chance_level=True
# )
# plt.title('SVM ROC Curve')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.legend()
# plt.savefig('ROC2.jpg')
#
# PrecisionRecallDisplay.from_predictions(
#     test_label,
#     test_pred,
#     plot_chance_level=True
# )
# plt.title('SVM PR Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.savefig('PR2.jpg')