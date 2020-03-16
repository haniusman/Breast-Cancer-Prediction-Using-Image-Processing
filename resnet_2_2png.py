import numpy as np
import pydicom as pdicom
import os
import glob
import cv2
import matplotlib.pyplot as plt

'exec(%matplotlib inline)'
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Dense,
    Flatten
)
from keras.applications.resnet50 import ResNet50
from keras.layers.merge import concatenate, add
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import activations
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
from keras import callbacks


def resnet_pseudo(img_dim=224, freeze_layers=10, full_freeze='N'):

    model = ResNet50(weights='imagenet', include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(4, activation='softmax')(x)

    model_final = Model(input=model.input, outputs=out)

    if full_freeze != 'N':
        for layer in model.layers[0:freeze_layers]:
            layer.trainable = False
    return model_final


def main():
    def load_scan2(path):
        print("load_scan2")
        list1 = []
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    list1.append(os.path.join(dirName, filename))
        return list1

    final = []

    a = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/MLO/benign-calc/*.png")]
    bclen = len(a)
    print(len(a), "bc-mlo")
    for i in range(0, len(a)):
        final.append(a[i])

    b = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/CC/benign-calc/*.png")]
    bclen = bclen + len(b)
    print(len(b), "bc-cc")
    for i in range(0, len(b)):
        final.append(b[i])

    c = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/MLO/benign-mass/*.png")]
    bmlen = len(c)
    print(len(c), "bm-mlo")
    for i in range(0, len(c)):
        final.append(c[i])
    d = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/CC/benign-mass/*.png")]
    bmlen = bmlen + len(d)
    print(len(d), "bm-cc")

    for i in range(0, len(d)):
        final.append(d[i])

    e = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/MLO/malignant-calc/*.png")]
    mclen = len(e)
    print(len(e), "mc-mlo")

    for i in range(0, len(e)):
        final.append(e[i])

    f = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/CC/malignant-calc/*.png")]
    mclen = mclen +len(f)
    print(len(f), "mc-cc")

    for i in range(0, len(f)):
        final.append(f[i])

    g = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/MLO/malignant-mass/*.png")]
    mmlen = len(g)
    print(len(g), "mm-mlo")

    for i in range(0, len(g)):
        final.append(g[i])

    h = [cv2.imread(file) for file in glob.glob("/home/genomics/PycharmProjects/BreastCancerPredictor/png-images/training/resized/CC/malignant-mass/*.png")]
    mmlen =mmlen + len(g)
    print(len(h),"mm-cc")

    for i in range(0, len(h)):
        final.append(h[i])


    print(bclen, bmlen, mclen, mmlen)
    x = len(final)
    print(x)
#
    final = np.array(final)

    num_classes = 4
    num_of_samples = len(final)

    labels = np.ones((num_of_samples,), dtype='int64')

    labels[0:bclen] = 0
    labels[bclen:bmlen] = 1
    labels[bmlen:mclen] = 2
    labels[mclen:mmlen] = 3

    print("labels")
    L = np_utils.to_categorical(labels, num_classes)
    del labels

    final, L = shuffle(final, L, random_state=2)
    print(final[0].shape)
    print("shuffle")
    train_x, val_x, train_y, val_y = train_test_split(final, L, stratify=L, test_size=0.2)
    print(train_x[0].shape)
    del L
    print("split")
    print(len(train_x),len(train_y),len(val_x),len(val_y))
    train_y = np.array(train_y)
    print("train_y")
    train_x = np.array(train_x)
    print(train_x[0].shape)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    print("model")

    # for i in range(0, len(train_x)):
    #     train_x[i] = train_x[i] / 255

    print("normalize")
    model = resnet_pseudo()
    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    filename = 'model_train_new2.csv'
    csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

    filepath = "/home/genomics/PycharmProjects/BreastCancerPredictor/try10_png.hdf5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [csv_log, early_stopping, checkpoint]


    hist = model.fit(train_x, train_y, batch_size=32, epochs=20, verbose=1, validation_data=(val_x, val_y), callbacks=callbacks_list)

    model.save('model10png2.hdf5')

if __name__ == '__main__':
    main()