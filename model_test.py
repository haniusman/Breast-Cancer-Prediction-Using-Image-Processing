import numpy as np
import pydicom as pdicom
import os
from sklearn.utils import shuffle
import cv2
import matplotlib.pyplot as plt
import glob
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
a = [cv2.imread(file) for file in glob.glob('/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/MLO/benign-calc/*.png')]
alen = len(a)
for i in range(0,alen):
    mlo.append(a[i])
print(alen)

h = [cv2.imread(file) for file in glob.glob('/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/CC/benign-calc/*.png')]
hlen = len(h)
for i in range(0, hlen):
    mlo.append(h[i])
print(hlen)

b = [cv2.imread(file) for file in glob.glob('/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/MLO/benign-mass/*.png')]
blen = len(b)
for i in range(0,blen):
    mlo.append(b[i])
print(blen)

f=[cv2.imread(file) for file in glob.glob('/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/CC/benign-mass/*.png')]
flen = len(f)
for i in range(0, flen):
    mlo.append(f[i])
print(flen)
e = [cv2.imread(file) for file in glob.glob(
    '/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/MLO/malignant-calc/*.png')]
elen = len(e)
for i in range(0, elen):
    mlo.append(e[i])
print(elen)

j = [cv2.imread(file) for file in glob.glob('/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/CC/malignant-calc/*.png')]
jlen = len(j)
for i in range(0, jlen):
    mlo.append(j[i])
print(hlen)

c = [cv2.imread(file) for file in glob.glob(
    '/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/MLO/malignant-mass/*.png')]
clen = len(c)
for i in range(0, clen):
    mlo.append(c[i])
print(clen)

g= [cv2.imread(file) for file in glob.glob('/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/testing/resized/CC/malignant-mass/*.png')]
glen = len(g)
for i in range(0, glen):
    mlo.append(g[i])

print(glen)
bc=alen+hlen
print('total')
print(bc)
bm=flen+blen
print(bm)
mc=jlen+elen
print(mc)
mm=clen+glen

print(mm)
print("mlo total")
print(len(mlo))
img = []

# for i in range(0, len(mlo)):
#     mlo[i] = np.expand_dims(mlo[i].pixel_array, axis=2)
# for i in range(0, len(mlo)):
#     img.append(mlo[i].pixel_array)
# final_1 = []
# for i in range(0, len(mlo)):
#     img[i] = img[i].astype(float)
#     final_1.append(np.stack((img[i],) * 3, axis=-1))

mlo=np.array(mlo)
num_classes = 4
num_of_samples = len(mlo)
mlolabels = np.ones((num_of_samples,), dtype='int64')

mlolabels[0:bc] = 0
mlolabels[bc:bm] = 1
mlolabels[bm:mc] = 2
mlolabels[mc:mm] = 3

loaded_model = load_model('/home/genomics/PycharmProjects/BreastCancerPredictor/model10png2.hdf5')


M = np_utils.to_categorical(mlolabels, num_classes)
mlo, M = shuffle(mlo, M, random_state=2)
mlo = np.array(mlo)
M = np.array(M)
e=np.array(e)
test_image = e[0:10]
print (test_image.shape)
print("predicted:")
print(loaded_model.predict(test_image))
print("actual class:")
print(M[1:2])


score = loaded_model.evaluate(mlo ,M, verbose=1)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

Y_pred = loaded_model.predict(mlo)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['class 1(benign-calcification)', 'class 2(benign-mass)', 'class 3(malignant-calcification)' , 'class4(malignant-mass']
print(classification_report(np.argmax(M, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(M,axis=1), y_pred))


#
# cnf_matrix = (confusion_matrix(np.argmax(M,axis=1), y_pred))
#
# np.set_printoptions(precision=2)
#
# plt.figure()

