import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate,\
    Conv2D, Add, Activation, Lambda, Conv1D, MaxPooling2D, UpSampling2D, Conv2DTranspose, AveragePooling2D, add
from Dropblock import *

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])

def ESA_Block(input_feature):
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        feature = input_feature
    f = channel // 4
    img_w, img_h = input_feature._keras_shape[1], input_feature._keras_shape[2]

    conv1 = Conv2D(f, (1, 1))(feature)
    stride_conv = Conv2D(f, (3, 3),  strides=(2, 2))(conv1)

    max_pool = MaxPooling2D(pool_size=(7, 7), strides=(3, 3))(stride_conv)

    conv_2 = Conv2D(f, (3, 3), padding='same', activation='relu')(max_pool)
    conv_21 = Conv2D(f, (3, 3), padding='same', activation='relu')(conv_2)
    conv_3 = Conv2D(f, (3, 3), padding='same')(conv_21)

    up = Lambda(my_upsampling, arguments={'img_w': img_w, 'img_h': img_h})(conv_3)
    conv4 = Conv2D(f, (1, 1))(conv1)

    connection = add([up, conv4])
    fea = Conv2D(channel, (1, 1), activation='sigmoid')(connection)

    return multiply([input_feature, fea])

def my_upsampling(x,img_w,img_h,method=0):
  """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
  return tf.image.resize_images(x, (img_w, img_h), method)
