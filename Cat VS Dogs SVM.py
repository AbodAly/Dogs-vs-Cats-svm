import os
from cv2 import resize
from skimage.feature import hog
import cv2
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import time


def create_Frame():
    train = pd.DataFrame({'file': os.listdir('train')})
    return train.iloc[:10000, :]


train = create_Frame()

Class = []
features = []

for i in os.listdir('train'):
    if 'dog' in i:
        Class.append(1)
        # Extracting Dog features
        # resized_img = resize(cv2.imread('train/' + i), (128, 64))
        # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
        #                     cells_per_block=(2, 2), visualize=True, multichannel=True)
        # features.append(fd)
    if len(Class) > 4999:
        break
count = 0
for i in os.listdir('train'):
    if 'cat' in i:
        Class.append(0)
        # Extracting Cat features
        # resized_img = resize(cv2.imread('train/' + i), (128, 64))
        # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
        #                 cells_per_block=(2, 2), visualize=True, multichannel=True)
        # features.append(fd)
    if len(Class) > 9999:
        break

train['Class'] = Class
# when reading data after
# pd.DataFrame(features).to_csv('Data_Features.csv', index=None)

x = pd.read_csv('Data_Features.csv')
y = train['Class']

Train, Test, Train_Y, Test_Y = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)

C = .01  # SVM regularization parameter

start = time.time()
svc = svm.SVC(kernel='linear', C=C).fit(Train, Train_Y)
stop = time.time()
Time = stop - start
print('SVC Linear Fitting Time = ', Time)
predictions = svc.predict(Test)
accuracy = np.mean(predictions == Test_Y)
print("acc = ", accuracy)

start = time.time()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(Train, Train_Y)
stop = time.time()
Time = stop - start
print('SVC Poly Fitting Time = ', Time)
predictions = poly_svc.predict(Test)
accuracy = np.mean(predictions == Test_Y)
print("acc = ", accuracy)