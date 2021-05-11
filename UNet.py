import math

from keras.optimizers import *
from keras.models import Model

from keras.layers import Input, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, UpSampling2D,\
    Conv2D, DepthwiseConv2D, Add, concatenate
import keras.backend as K
from Spatial_Attention import *

def ESA_UNet(input_size=(512, 512, 3), block_size=7,keep_prob=0.9,start_neurons=16, lr=1e-3):

    inputs = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(conv1)
    conv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(conv2)
    conv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(conv3)
    conv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)
    convm = ESA_Block(convm)
    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(convm)
    convm = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    return model


def MSFF_Net(input_size=(512, 512, 3), block_size=7,keep_prob=0.9,start_neurons=16, lr=1e-3):

    inputs = Input(input_size)
    conv11 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv11)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(conv1)
    conv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = spatial_attention(conv1)
    conv1 = Add()([conv11, conv1])
    # conv1 = channel_attention(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv21 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv21)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(conv2)
    conv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = spatial_attention(conv2)
    conv2 = Add()([conv21, conv2])
    # conv2 = channel_attention(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv31 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv31)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(conv3)
    conv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    # conv3 = channel_attention(conv3)
    conv3 = spatial_attention(conv3)
    conv3 = Add()([conv31, conv3])
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm1 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm1)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)

    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(convm)
    convm = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)
    convm = spatial_attention(convm)
    convm = Add()([convm1, convm])
    # convm = channel_attention(convm)

    """
    # deconv3 = Lambda(my_upsampling, arguments={'img_w': conv3._keras_shape[1], 'img_h': conv3._keras_shape[2]})(convm)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)

    # deconv2 = Lambda(my_upsampling, arguments={'img_w': conv2._keras_shape[1], 'img_h': conv2._keras_shape[2]})(uconv3)
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)

    # deconv1 = Lambda(my_upsampling, arguments={'img_w': conv1._keras_shape[1], 'img_h': conv1._keras_shape[2]})(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    """
    # """
    # deconv3 = Lambda(my_upsampling, arguments={'img_w': conv1._keras_shape[1], 'img_h': conv1._keras_shape[2]})(convm)
    # deconv2 = Lambda(my_upsampling, arguments={'img_w': conv1._keras_shape[1], 'img_h': conv1._keras_shape[2]})(conv3)
    # deconv1 = Lambda(my_upsampling, arguments={'img_w': conv1._keras_shape[1], 'img_h': conv1._keras_shape[2]})(conv2)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(8, 8), padding="same")(convm)
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(4, 4), padding="same")(conv3)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(conv2)



    conv = Lambda(concat)([deconv3, deconv2, deconv1, conv1])

    uconv11 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(conv)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv11)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    # """

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)


    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    return model


def ODE_UNet(input_size=(512, 512, 3), block_size=7,keep_prob=0.9,start_neurons=16, lr=1e-3):

    """
    inputs = Input(input_size)
    conv11 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", dilation_rate=2)(inputs)
    conv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv11)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same", dilation_rate=4)(conv1)
    conv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # conv1 = spatial_attention(conv1)
    conv1 = Add()([conv11, conv1])
    # conv1 = channel_attention(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv21 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", dilation_rate=2)(pool1)
    conv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv21)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same", dilation_rate=4)(conv2)
    conv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    # conv2 = spatial_attention(conv2)
    conv2 = Add()([conv21, conv2])
    # conv2 = channel_attention(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv31 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", dilation_rate=2)(pool2)
    conv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv31)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same", dilation_rate=4)(conv3)
    conv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    # conv3 = channel_attention(conv3)
    # conv3 = spatial_attention(conv3)
    conv3 = Add()([conv31, conv3])
    pool3 = MaxPooling2D((2, 2))(conv3)
    """

    inputs = Input(input_size)
    conv1 = LF_m(inputs, start_neurons * 1)
    # conv1 = RK2_m(inputs, start_neurons * 1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = LF_m(pool1, start_neurons * 2)
    # conv2 = RK2_m(pool1, start_neurons * 2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = LF_m(pool2, start_neurons * 4)
    # conv3 = RK2_m(pool2, start_neurons * 4)
    pool3 = MaxPooling2D((2, 2))(conv3)


    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)

    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(convm)
    convm = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    return model


def LF_m(input, out_channels, block_size=7, keep_prob=0.9):
    yn = Conv2D(out_channels, (1, 1), activation=None, padding="same")(input)

    G_yn = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=1)(input)
    # G_yn = Conv2D(out_channels, (3, 3), activation=None, padding="same")(input)
    G_yn = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(G_yn)
    G_yn = BatchNormalization()(G_yn)
    yn_1 = Activation('relu')(G_yn)
    scale_1 = Lambda(lambda x: x * K.variable(2))
    yn_1 = scale_1(yn_1)

    Gyn_1 = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=2)(yn_1)
    # Gyn_1 = Conv2D(out_channels, (3, 3), activation=None, padding="same")(yn_1)
    Gyn_1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(Gyn_1)
    Gyn_1 = BatchNormalization()(Gyn_1)
    yn_2 = Activation('relu')(Gyn_1)
    scale_2 = Lambda(lambda x: x * K.variable(2))
    yn_2 = scale_2(yn_2)
    yn_2 = Add()([yn_2, yn])

    Gyn_2 = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=4)(yn_2)
    # Gyn_2 = Conv2D(out_channels, (3, 3), activation=None, padding="same")(yn_2)
    Gyn_2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(Gyn_2)
    Gyn_2 = BatchNormalization()(Gyn_2)
    yn_3 = Activation('relu')(Gyn_2)
    scale_3 = Lambda(lambda x: x * K.variable(2))
    yn_3 = scale_3(yn_3)
    yn_3 = Add()([yn_3, yn_1])

    Gyn_3 = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=8)(yn_3)
    # Gyn_3 = Conv2D(out_channels, (3, 3), activation=None, padding="same")(yn_3)
    Gyn_3 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(Gyn_3)
    Gyn_3 = BatchNormalization()(Gyn_3)
    yn_4 = Activation('relu')(Gyn_3)
    scale_3 = Lambda(lambda x: x * K.variable(2))
    yn_4 = scale_3(yn_4)
    output = Add()([yn_4, yn_2])

    return output


def RK2_m(input, out_channels, block_size=7, keep_prob=0.9):
    yn = Conv2D(out_channels, (1, 1), activation=None, padding="same")(input)
    # G_yn = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=2)(input)
    G_yn = Conv2D(out_channels, (3, 3), activation=None, padding="same")(input)
    G_yn = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(G_yn)
    G_yn = BatchNormalization()(G_yn)
    G_yn = Activation('relu')(G_yn)

    yn_1 = Add()([G_yn, yn])

    # Gyn_1 = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=4)(yn_1)
    Gyn_1 = Conv2D(out_channels, (3, 3), activation=None, padding="same")(yn_1)
    Gyn_1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(Gyn_1)
    Gyn_1 = BatchNormalization()(Gyn_1)
    Gyn_1 = Activation('relu')(Gyn_1)

    yn_2 = Add()([Gyn_1, G_yn])
    scale_1 = Lambda(lambda x: x * K.variable(0.5))
    yn_2 = scale_1(yn_2)

    output = Add()([yn_2, yn])

    return output


def RK3(input, out_channels, block_size=7, keep_prob=0.9):
    yn = Conv2D(out_channels, (1, 1), activation=None, padding="same")(input)
    G_yn = Conv2D(out_channels, (3, 3), activation=None, padding="same")(input)
    # G_yn = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(G_yn)
    G_yn = BatchNormalization()(G_yn)
    G_yn = Activation('relu')(G_yn)
    scale_1 = Lambda(lambda x: x * K.variable(0.5))
    yn_1 = scale_1(G_yn)
    yn_1 = Add()([yn_1, yn])

    Gyn_1 = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=2)(yn_1)
    # Gyn_1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(Gyn_1)
    Gyn_1 = BatchNormalization()(Gyn_1)
    Gyn_1 = Activation('relu')(Gyn_1)
    scale_2 = Lambda(lambda x: x * K.variable(2))
    yn_2 = scale_2(Gyn_1)
    yn_2 = Add()([yn, yn_2])
    scale_3 = Lambda(lambda x: x * K.variable(-1.0))
    temp = scale_3(G_yn)
    yn_2 = Add()([yn_2, temp])

    Gyn_2 = Conv2D(out_channels, (3, 3), activation=None, padding="same", dilation_rate=4)(yn_2)
    # Gyn_2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)(Gyn_2)
    Gyn_2 = BatchNormalization()(Gyn_2)
    Gyn_2 = Activation('relu')(Gyn_2)
    scale_4 = Lambda(lambda x: x * K.variable(4))
    yn_3 = scale_4(Gyn_1)
    yn_3 = Add()([Gyn_2, yn_3, G_yn])
    scale_5 = Lambda(lambda x: x * K.variable(1/6))
    yn_3 = scale_5(yn_3)
    output = Add()([yn_3, yn])

    return output


def my_upsampling(x,img_w,img_h,method=0):
  """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""

  return tf.image.resize_images(x, (img_w, img_h), method)

def concat(x = []):
    return K.concatenate(x)