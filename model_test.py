import numpy as np
import pydicom as pdicom
import os
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
from keras import backend as K
#K.set_image_dim_ordering('th')
from keras import models
from keras.models import load_model
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam



num_channel=1
def load_scan2(path):
    list1 = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                list1.append(os.path.join(dirName, filename))

    return list1
mlo = []
cc = []
a = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/MLO/benign_calc')
alen = len(a)
b = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/MLO/benign_mass')
blen = len(b)
c = load_scan2(
    '/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/MLO/malignant_mass')
clen = len(c)
e = load_scan2(
    '/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/MLO/malignant_calc')
elen = len(e)

f= load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/CC/benign_mass')
flen = len(f)
g= load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/CC/malignant_mass')
glen = len(g)

h =  load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/CC/benign_calc')
hlen = len(h)
j = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/Testing Images/resize/CC/malignant_calc')
jlen = len(j)

# bc
for i in range(0, 20):
    mlo.append(pdicom.read_file(a[i]))
for i in range(0, 20):
    mlo.append(pdicom.read_file(h[i]))
#bm
for i in range(0, 20):
    mlo.append(pdicom.read_file(b[i]))
for i in range(0, 20):
    mlo.append(pdicom.read_file(f[i]))

# mc
for i in range(0, 20):
    mlo.append(pdicom.read_file(c[i]))
for i in range(0, 20):
    mlo.append(pdicom.read_file(j[i]))
# mm
for i in range(0, 20):
    mlo.append(pdicom.read_file(e[i]))
for i in range(0, 20):
    mlo.append(pdicom.read_file(g[i]))

print(len(mlo))
img = []

# for i in range(0, len(mlo)):
#     mlo[i] = np.expand_dims(mlo[i].pixel_array, axis=2)
for i in range(0, len(mlo)):
    img.append(mlo[i].pixel_array)
final_1 = []
for i in range(0, len(mlo)):
    img[i] = img[i].astype(float)
    final_1.append(np.stack((img[i],) * 3, axis=-1))

num_classes = 4
num_of_samples = len(final_1)
mlolabels = np.ones((num_of_samples,), dtype='int64')

mlolabels[0:40] = 0
mlolabels[40:80] = 1
mlolabels[80:120] = 2
mlolabels[120:160] = 3

loaded_model = load_model('/home/genomics/PycharmProjects/BreastCancerPredictor/model10.hdf5')


M = np_utils.to_categorical(mlolabels, num_classes)
mlo, M = shuffle(final_1, M, random_state=2)
mlo = np.array(mlo)
M = np.array(M)

# test_image = mlo[1:2]
# print (test_image.shape)
# print("predicted:")
# print(loaded_model.predict(test_image))
# print("actual class:")
# print(M[1:2])

#
# score = loaded_model.evaluate(mlo ,M, verbose=1)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])

Y_pred = loaded_model.predict(mlo)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['class 0(benign-calcification)', 'class 1(benign-mass)', 'class 2(malignant-calcification)', 'class 3(malignant-mass)']

print(classification_report(np.argmax(M, axis=1), y_pred, target_names=target_names))

print(confusion_matrix(np.argmax(M,axis=1), y_pred))



cnf_matrix = (confusion_matrix(np.argmax(M,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

