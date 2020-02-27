import numpy as np
import pydicom as pdicom
import os
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

    bc = []
    bm = []
    mm = []
    mc = []

    img = []
    final = []
    final_1 = []


    a = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/benign-calc')
    for i in range(0, 196):
        # bc.append(pdicom.read_file(a[i]))
        final.append(pdicom.read_file(a[i]))
    bclen = len(a)

    a = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/benign-calc')
    for i in range(0, 196):
        # bc.append(pdicom.read_file(a[i]))
        final.append(pdicom.read_file(a[i]))
    bclen = bclen + len(a)

    b = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/benign-mass')
    for i in range(0, 196):
        # bm.append(pdicom.read_file(b[i]))
        final.append(pdicom.read_file(b[i]))

    bmlen = len(b)

    b = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/benign-mass')
    for i in range(0, 196):
        # bm.append(pdicom.read_file(b[i]))
        final.append(pdicom.read_file(b[i]))

    bmlen = bmlen + len(b)

    c = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/malignant-mass')
    for i in range(0, 196):
        # mm.append(pdicom.read_file(c[i]))
        final.append(pdicom.read_file(c[i]))

    mmlen = len(c)

    # c = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/malignant-mass')
    for i in range(0, 196):
        # mm.append(pdicom.read_file(c[i]))
        final.append(pdicom.read_file(c[i]))

    mmlen = mmlen + len(c)

    e = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/malignant-calc')
    for i in range(0, 196):
        # mc.append(pdicom.read_file(e[i]))
        final.append(pdicom.read_file(e[i]))
    mclen = len(e)

    e = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/malignant-calc')
    for i in range(0, 196):
        # mc.append(pdicom.read_file(e[i]))
        final.append(pdicom.read_file(e[i]))
    mclen = mclen + len(e)

    print(len(final))

    for i in range(0, len(final)):
        img.append(final[i].pixel_array)

    for i in range(0, 500):
        img[i] = img[i].astype(float)
        final_1.append(np.stack((img[i],) * 3, axis=-1))

    for i in range(500, 1000):
        img[i] = img[i].astype(float)
        final_1.append(np.stack((img[i],) * 3, axis=-1))

    for i in range(1000, 1568):
        img[i] = img[i].astype(float)
        final_1.append(np.stack((img[i],) * 3, axis=-1))

    final_1 = np.array(final_1)
    print(final_1[0].shape)

    num_classes = 4
    num_of_samples = len(final_1)
    labels = np.ones((num_of_samples,), dtype='int64')

    labels[0:392] = 0
    labels[392:784] = 1
    labels[784:1176] = 2
    labels[1176:1568] = 3

    print("labels")
    L = np_utils.to_categorical(labels, num_classes)

    final_1, L = shuffle(final_1, L, random_state=2)
    print("shuffle")
    train_x, val_x, train_y, val_y = train_test_split(final_1, L, stratify=L, test_size=0.2)
    print("split")
    print(len(train_x),len(train_y),len(val_x),len(val_y))
    train_y = np.array(train_y)
    print("train_y")
    train_x = np.array(train_x)
    print("train_x")
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    print("model")

    for i in range(0, len(train_x)):
        train_x[i] = train_x[i] / 255

    print("normalize")
    model = resnet_pseudo()
    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    filename = 'model_train_new2.csv'
    csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

    filepath = "/home/genomics/PycharmProjects/BreastCancerPredictor/try10.hdf5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [csv_log, early_stopping, checkpoint]


    hist = model.fit(train_x, train_y, batch_size=10, epochs=4, verbose=1, validation_data=(val_x, val_y), callbacks=callbacks_list)

    model.save('model10.hdf5')

if __name__ == '__main__':
    main()