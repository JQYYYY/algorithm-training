import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature as sk
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC


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
    train_data = np.zeros((160, 256, 256))
    test_data = np.zeros((40, 256, 256))
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (256, 256))
        test_data[test_index, :, :] = image
        test_index += 1
    # print(train_label)
    train_data = train_data.astype(np.uint8)
    test_data = test_data.astype(np.uint8)
    return train_data, train_label, test_data, test_label


# LBP特征提取
def texture_detect(train_data, test_data):
    radius = 1      # 2
    n_point = radius * 8
    train_hist = np.zeros((160, 256))
    test_hist = np.zeros((40, 256))
    for i in np.arange(160):
        lbp = sk.local_binary_pattern(train_data[i], n_point, radius, 'default')
        max_bins = int(lbp.max() + 1)
        train_hist[i], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))

    for i in np.arange(40):
        lbp = sk.local_binary_pattern(test_data[i], n_point, radius, 'default')
        max_bins = int(lbp.max() + 1)
        test_hist[i], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
    return train_hist, test_hist


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
    plt.savefig(s + '.jpg')
    # plt.show()


train_data, train_label, test_data, test_label = load_image()
train_hist, test_hist = texture_detect(train_data, test_data)
train_label = np.array(train_label)
test_label = np.array(test_label)

# bins = np.array([i for i in range(256)])
# plt.bar(bins, train_hist[0], width=1, label='Airplane')
# plt.bar(bins, train_hist[1], width=1, label='Forest')
# plt.xlabel('Bins')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.legend()
# plt.savefig('hist.jpg')

# 可视化调参
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
# figure_in_sample(test_score, 'test')

# 预测测试集
estimator = SVC(kernel='rbf', C=10, gamma=80)
estimator.fit(train_hist, train_label)
test_pred = estimator.predict(test_hist)
score = estimator.score(test_hist, test_label)
print(score)
print(test_pred)
print(test_label)

# 混淆矩阵
cm = confusion_matrix(test_label, test_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# ROC曲线
# RocCurveDisplay.from_predictions(
#     test_label,
#     test_pred,
#     plot_chance_level=True
# )
# plt.title('LBP-SVM ROC Curve')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.legend()
# plt.savefig('ROC1.jpg')
#
# PrecisionRecallDisplay.from_predictions(
#     test_label,
#     test_pred,
#     plot_chance_level=True
# )
# plt.title('LBP-SVM PR Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.savefig('PR1.jpg')