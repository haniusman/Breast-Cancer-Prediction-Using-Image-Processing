import numpy as np
import pydicom as pdicom
import os
import matplotlib.pyplot as plt

'exec(%matplotlib inline)'
from keras.utils import np_utils
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Dense,
    Flatten
)
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
# import warnings
# warnings.filterwarnings('error')

if K.image_data_format() == 'channels_last':
    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
else:
    CHANNEL_AXIS = 1
    ROW_AXIS = 2
COL_AXIS = 3

# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions,
                    is_first_layer=False, shortcut_with_bn=False,
                    bottleneck_enlarge_factor=4, **kw_args):
    print("residual-block")
    def f(input):
        print("f-residual")
        for i in range(repetitions):
            init_strides = (1, 1)
            identity = True
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            if i == 0:
                identity = False
            input = block_function(nb_filters=nb_filters,
                                   init_strides=init_strides,
                                   identity=identity,
                                   shortcut_with_bn=shortcut_with_bn,
                                   enlarge_factor=bottleneck_enlarge_factor,
                                   **kw_args)(input)
        return input

    return f
def _conv_bn_relu(nb_filter, nb_row, nb_col, strides=(1, 1),
                  weight_decay=.0001, dropout=.0, last_block=False):
    print("_conv_bn_relu")
    def f(input):
        print("f-conv")

        # conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
        #               strides=strides, kernel_initializer="he_normal",
        #               padding="same", kernel_regularizer=l2(weight_decay))(input)
        conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
                      strides=strides, kernel_initializer="he_normal",
                      padding="same", kernel_regularizer=None)(input)


        norm = BatchNormalization(axis=CHANNEL_AXIS)(conv)
        if last_block:
            return norm
        else:
            relu = Activation("relu")(norm)
            return Dropout(dropout)(relu)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, strides=(1, 1),
                  weight_decay=.0001, dropout=.0):
    print("_bn_relu_conv")
    def f(input):
        print("f-bn-relu")
        norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        activation = Dropout(dropout)(activation)
        # return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
        #               strides=strides, kernel_initializer="he_normal",
        #               padding="same",
        #               kernel_regularizer=l2(weight_decay))(activation)
        return Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
                      strides=strides, kernel_initializer="he_normal",
                      padding="same",
                      kernel_regularizer=None)(activation)

    return f
def _shortcut(input, residual, weight_decay=.0001, dropout=.0, identity=True,
              strides=(1, 1), with_bn=False, org=False):
    print("Shortcut")
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    # !!! The dropout argument is just a place holder.
    # !!! It shall not be applied to identity mapping.
    # stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    # stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    # equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_width > 1 or stride_height > 1 or not equal_channels:
    if not identity:
        #
        shortcut = Conv2D(filters=residual._keras_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1), strides=strides,
                          kernel_initializer="he_normal", padding="valid",
                          kernel_regularizer=None)(input)
        if with_bn:
            shortcut = BatchNormalization(axis=CHANNEL_AXIS)(shortcut)

    addition = add([shortcut, residual])
    if not org:
        return addition
    else:
        relu = Activation("relu")(addition)
    return Dropout(dropout)(relu)

def bottleneck(nb_filters, init_strides=(1, 1), identity=True,
               shortcut_with_bn=False, enlarge_factor=4, **kw_args):
    print("Bottleneck")
    def f(input):
        print("f-bottleneck")
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, strides=init_strides, **kw_args)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3, **kw_args)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * enlarge_factor, 1, 1, **kw_args)(conv_3_3)
        return _shortcut(input, residual, identity=identity,
                         strides=init_strides,
                         with_bn=shortcut_with_bn, **kw_args)

    return f

class ResNetBuilder(object):
    @staticmethod
    def _shared_conv_layers(input_shape, block_fn, repetitions, nb_init_filter=64,
                            init_filter_size=7, init_conv_stride=2, pool_size=3,
                            pool_stride=2,
                            weight_decay=.0001, inp_dropout=.0, hidden_dropout=.0,
                            shortcut_with_bn=False,
                            bottleneck_enlarge_factor=4 ,view="def"):
        print("_shared_conv_layers")
        print(view)

        '''Create shared conv layers for all inputs
        Args:
            pool_size ([int]): set to 0 or False to turn off the first max pooling.
        '''

        # if len(input_shape) != 3:
        #     raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input_ = Input(shape=input_shape)
        dropped = Dropout(inp_dropout)(input_)
        conv1 = _conv_bn_relu(nb_filter=nb_init_filter,
                              nb_row=init_filter_size,
                              nb_col=init_filter_size,
                              strides=(init_conv_stride, init_conv_stride),
                              weight_decay=weight_decay, dropout=hidden_dropout)(dropped)
        if pool_size:
            pool1 = MaxPooling2D(pool_size=(pool_size, pool_size),
                                 strides=(pool_stride, pool_stride),
                                 padding="same")(conv1)
            block = pool1
        else:
            block = conv1

        nb_filters = nb_init_filter
        for i, r in enumerate(repetitions):
            block = _residual_block(
                block_fn, nb_filters=nb_filters, repetitions=r,
                is_first_layer=(i == 0),
                shortcut_with_bn=shortcut_with_bn,
                bottleneck_enlarge_factor=bottleneck_enlarge_factor,
                weight_decay=weight_decay,
                dropout=hidden_dropout)(block)
            nb_filters *= 2

        # Classifier block
        pool2 = GlobalAveragePooling2D()(block)

        return input_, pool2

    @staticmethod
    def l1l2_penalty_reg(alpha=1.0, l1_ratio=0.5):
        print("l1l2 penalty reg")
        '''Calculate L1 and L2 penalties for a Keras layer
        This follows the same formulation as in the R package glmnet and Sklearn
        Args:
            alpha ([float]): amount of regularization.
            l1_ratio ([float]): portion of L1 penalty. Setting to 1.0 equals
                    Lasso.
        '''
        if l1_ratio == .0:
            return l2(alpha)
        elif l1_ratio == 1.:
            return l1(alpha)
        else:
            return l1_l2(l1_ratio*alpha, 1./2*(1 - l1_ratio)*alpha)

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, nb_init_filter=64,
              init_filter_size=7, init_conv_stride=2, pool_size=3, pool_stride=2,
              weight_decay=.0001, alpha=1., l1_ratio=.5,
              inp_dropout=.0, hidden_dropout=.0, shortcut_with_bn=False):
        print("build-1")

        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        :param num_outputs: The number of outputs at final softmax layer
        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50
        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved
        :return: The keras model.
        """

        inputs, flatten_out = ResNetBuilder._shared_conv_layers(
            input_shape, block_fn, repetitions,
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay,
            inp_dropout=inp_dropout, hidden_dropout=hidden_dropout,
            shortcut_with_bn=shortcut_with_bn)
        enet_penalty = ResNetBuilder.l1l2_penalty_reg(alpha, l1_ratio)
        activation = "softmax" if num_outputs > 1 else "sigmoid"
        #
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation=activation, kernel_regularizer=None)(flatten_out)
        model = Model(inputs=inputs, outputs=dense)
        return model


    @classmethod
    def build_resnet_50(cls, input_shape, num_outputs,
                        nb_init_filter=64, init_filter_size=7, init_conv_stride=2,
                        pool_size=3, pool_stride=2,
                        weight_decay=.0001, alpha=1., l1_ratio=.5,
                        inp_dropout=.0, hidden_dropout=.0,
                        shortcut_with_bn=False):

        print("build_resnet-50")
        return cls.build(
            input_shape, num_outputs, bottleneck, [2, 2, 2, 2],
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay, inp_dropout=inp_dropout,
            hidden_dropout=hidden_dropout, shortcut_with_bn=shortcut_with_bn)


class MultiViewResNetBuilder(ResNetBuilder):
    '''Residual net with two inputs
    '''
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, nb_init_filter=64,
              init_filter_size=7, init_conv_stride=2, pool_size=3, pool_stride=2,
              weight_decay=.0001, alpha=1., l1_ratio=.5,
              inp_dropout=.0, hidden_dropout=.0, shortcut_with_bn=False):
        print("build")

        """
        Builds a custom ResNet like architecture.
        :param input_shape: Shall be the input shapes for both CC and MLO views.
        :param num_outputs: The number of outputs at final softmax layer
        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50
        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved
        :return: The keras model.
        """

        # First, define a shared CNN model for both CC and MLO views.
        input_cc, flatten_cc = ResNetBuilder._shared_conv_layers(
            input_shape, block_fn, repetitions,
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay,
            inp_dropout=inp_dropout, hidden_dropout=hidden_dropout,
            shortcut_with_bn=shortcut_with_bn, view="cc")


        input_mlo, flatten_mlo = ResNetBuilder._shared_conv_layers(
            input_shape, block_fn, repetitions,
            nb_init_filter=nb_init_filter, init_filter_size=init_filter_size,
            init_conv_stride=init_conv_stride,
            pool_size=pool_size, pool_stride=pool_stride,
            weight_decay=weight_decay,
            inp_dropout=inp_dropout, hidden_dropout=hidden_dropout,
            shortcut_with_bn=shortcut_with_bn, view="mlo")


        # Then merge the conv representations of the two views.
        merged_repr = concatenate([flatten_cc, flatten_mlo])
        enet_penalty = ResNetBuilder.l1l2_penalty_reg(alpha, l1_ratio)
        activation = "softmax" if num_outputs > 1 else "sigmoid"
        #
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation=activation, kernel_regularizer=None)(merged_repr)
        discr_model = Model(inputs=[input_cc, input_mlo], outputs=dense)
        return discr_model

def main():
    def load_scan2(path):
        print("load_scan2")
        list1 = []
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    list1.append(os.path.join(dirName, filename))

        return list1

    mlo = []
    a = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/benign-calc')
    alen = len(a)
    b = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/benign-mass')
    blen = len(b)
    c = load_scan2(
        '/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/malignant-mass')
    clen = len(c)
    e = load_scan2(
        '/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/MLO/malignant-calc')
    elen = len(e)
    for i in range(0, 464):
        mlo.append(pdicom.read_file(a[i]))
    for i in range(0, 196):
        mlo.append(pdicom.read_file(b[i]))
    for i in range(0, 215):
        mlo.append(pdicom.read_file(c[i]))
    for i in range(0, 254):
        mlo.append(pdicom.read_file(e[i]))
    # print(alen)
    # print(blen)
    # print(clen)
    # print(elen)
    print(len(mlo))

    for i in range(0, len(mlo)):
        mlo[i] = np.expand_dims(mlo[i].pixel_array, axis=2)

    num_classes = 4
    num_of_samples = len(mlo)
    mlolabels = np.ones((num_of_samples,), dtype='int64')

    mlolabels[0:464] = 0
    mlolabels[464:660] = 1
    mlolabels[660:875] = 2
    mlolabels[875:1129] = 3

    M = np_utils.to_categorical(mlolabels, num_classes)
    # mlo, M = shuffle(mlo, M, random_state=2)


    # TO LOAD DATA FOR CC VIEW
    cc = []
    a = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/benign-calc')
    alen = len(a)
    b = load_scan2('/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/benign-mass')
    blen = len(b)
    c = load_scan2(
        '/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/malignant-mass')
    clen = len(c)
    e = load_scan2(
        '/home/genomics/PycharmProjects/BreastCancerPredictor/processed images-training/resized/CC/malignant-calc')
    elen = len(e)
    for i in range(0, 464):
        cc.append(pdicom.read_file(a[i]))
    for i in range(0, 196):
        cc.append(pdicom.read_file(b[i]))
    for i in range(0, 215):
        cc.append(pdicom.read_file(c[i]))
    for i in range(0, 254):
        cc.append(pdicom.read_file(e[i]))
    # print(alen)
    # print(blen)
    # print(clen)
    # print(elen)
    print(len(cc))

    for i in range(0, len(cc)):
        cc[i] = np.expand_dims(cc[i].pixel_array, axis=2)

    num_classes = 4
    num_of_samples = len(cc)
    cclabels = np.ones((num_of_samples,), dtype='int64')

    # cclabels[0:alen] = 0
    # cclabels[alen:blen] = 1
    # cclabels[blen:clen] = 2
    # cclabels[clen:elen] = 3

    cclabels[0:464] = 0
    cclabels[464:660] = 1
    cclabels[660:875] = 2
    cclabels[875:1129] = 3

    C = np_utils.to_categorical(cclabels, num_classes)
    cc, C,mlo = shuffle(cc, C, mlo, random_state=2)
    # print(cc.shape)
    mlo = np.array(mlo)
    M = np.array(M)
    cc = np.array(cc)
    print(cc.shape)
    C = np.array(C)
    for i in range(0,len(cc)):
      cc[i]=cc[i]/255
      mlo[i]=mlo[i]/255
    model = MultiViewResNetBuilder.build_resnet_50((1, 512, 512), 4, inp_dropout=.2, hidden_dropout=.5)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    #model.summary()
    # h=[]
    # for i in range(0,2):
    #     h.append(x[i])
    #
    # h = np.array(h)
    # print(h.shape)

    filename = 'model_train_new.csv'
    csv_log = callbacks.CSVLogger(filename, separator=',', append=False)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

    filepath = "/home/genomics/PycharmProjects/BreastCancerPredictor/try15.hdf5"

    checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [csv_log, early_stopping, checkpoint]



    hist = model.fit([cc, mlo], M, batch_size=10, epochs=4, verbose=1, validation_split=0.1, callbacks=callbacks_list)

    model.save('model15.hdf5')
    # loaded_model = load_model('model.hdf5')

    # visualizing losses and accuracy
    # train_loss = hist.history['loss']
    # val_loss = hist.history['val_loss']
    # train_acc = hist.history['accuracy']
    # val_acc = hist.history['val_accuracy']
    # xc = range(3)
    #
    #
    # plt.figure(1, figsize=(7, 5))
    # plt.plot(xc, train_loss)
    # plt.plot(xc, val_loss)
    # plt.xlabel('num of Epochs')
    # plt.ylabel('loss')
    # plt.title('train_loss vs val_loss')
    # plt.grid(True)
    # plt.legend(['train', 'val'])
    # # print plt.style.available # use bmh, classic,ggplot for big pictures
    # plt.style.use(['classic'])
    #
    #
    # plt.figure(2,figsize=(7,5))
    # plt.plot(xc,train_acc)
    # plt.plot(xc,val_acc)
    # plt.xlabel('num of Epochs')
    # plt.ylabel('accuracy')
    # plt.title('train_acc vs val_acc')
    # plt.grid(True)
    # plt.legend(['train', 'val'], loc=4)
    # # print plt.style.available # use bmh, classic,ggplot for big pictures
    # plt.style.use(['classic'])

#
#     # //Evaluating the model
# score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])
#
# test_image = X_test[0:1]
# print(test_image.shape)
#
# print(model.predict(test_image))
# print(model.predict_classes(test_image))
# print(y_test[0:1])
#
# # Testing a new image
# test_image = cv2.imread('data/Humans/rider-8.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# test_image = cv2.resize(test_image, (128, 128))
# test_image = np.array(test_image)
# test_image = test_image.astype('float32')
# test_image /= 255
# print(test_image.shape)
#
# if num_channel == 1:
#     if K.image_dim_ordering() == 'th':
#         test_image = np.expand_dims(test_image, axis=0)
#         test_image = np.expand_dims(test_image, axis=0)
#         print(test_image.shape)
#     else:
#         test_image = np.expand_dims(test_image, axis=3)
#         test_image = np.expand_dims(test_image, axis=0)
#         print(test_image.shape)
#
# else:
#     if K.image_dim_ordering() == 'th':
#         test_image = np.rollaxis(test_image, 2, 0)
#         test_image = np.expand_dims(test_image, axis=0)
#         print(test_image.shape)
#     else:
#         test_image = np.expand_dims(test_image, axis=0)
#         print(test_image.shape)
#
# # Predicting the test image
# print((model.predict(test_image)))
# print(model.predict_classes(test_image))
#


if __name__ == '__main__':
     main()
