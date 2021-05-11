import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras import losses
from scipy.misc.pilutil import *
from keras.optimizers import Adam, K
data_location = './'

training_images_loc = data_location + 'DRIVE/train/images/'
training_label_loc = data_location + 'DRIVE/train/labels/'

validate_images_loc = data_location + 'DRIVE/validate/images/'
validate_label_loc = data_location + 'DRIVE/validate/labels/'
train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []
desired_size = 592
for i in train_files:
    im = imread(training_images_loc + i)
    label = imread(training_label_loc + i.split('_')[0] + '_manual1.png', mode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)


for i in validate_files:
    im = imread(validate_images_loc + i)
    label = imread(validate_label_loc + i.split('_')[0] + '_manual1.png', mode="L")
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    validate_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    validate_label.append(temp)

train_data = np.array(train_data)
train_label = np.array(train_label)

validate_data = np.array(validate_data)
validate_label = np.array(validate_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (
len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

x_validate = validate_data.astype('float32') / 255.
y_validate = validate_label.astype('float32') / 255.
x_validate = np.reshape(x_validate, (
len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_validate = np.reshape(y_validate,
                        (len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)


from UNet import ESA_UNet
from UNet import MSFF_Net
model = MSFF_Net(input_size=(desired_size, desired_size, 3), start_neurons=16, lr=1e-3, keep_prob=0.82, block_size=7)
adam = Adam(lr=0.001, epsilon=1e-8)

def total_loss(y_true, y_pred):
    """dice loss function for tensorflow/keras
        calculate dice loss per batch and channel of each sample.
    Args:
        data_format: either channels_first or channels_last
    Returns:
        loss_function(y_true, y_pred)
    """
    if K.image_data_format() == "channels_last":
        y_pred = tf.transpose(y_pred, (0, 3, 1, 2))
        y_true = tf.transpose(y_true, (0, 3, 1, 2))

    smooth = 1.0
    iflat = tf.reshape(
        y_pred, (tf.shape(y_pred)[0], tf.shape(y_pred)[1], -1)
    )  # batch, channel, -1
    tflat = tf.reshape(y_true, (tf.shape(y_true)[0], tf.shape(y_true)[1], -1))
    intersection = K.sum(iflat * tflat, axis=-1)
    loss = 1 - ((2.0 * intersection + smooth)) / (
            K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth
    )

    return 0.2 * loss + K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def dice_loss(y_true, y_pred):
    """dice loss function for tensorflow/keras
        calculate dice loss per batch and channel of each sample.
    Args:
        data_format: either channels_first or channels_last
    Returns:
        loss_function(y_true, y_pred)
    """
    if K.image_data_format() == "channels_last":
        y_pred = tf.transpose(y_pred, (0, 3, 1, 2))
        y_true = tf.transpose(y_true, (0, 3, 1, 2))

    smooth = 1.0
    iflat = tf.reshape(
        y_pred, (tf.shape(y_pred)[0], tf.shape(y_pred)[1], -1)
    )  # batch, channel, -1
    tflat = tf.reshape(y_true, (tf.shape(y_true)[0], tf.shape(y_true)[1], -1))
    intersection = K.sum(iflat * tflat, axis=-1)
    loss = 1 - ((2.0 * intersection + smooth)) / (
            K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth
    )

    return loss

# model.compile(loss=dice_loss, optimizer=adam, metrics=['accuracy'])
# model.compile(loss=losses.binary_crossentropy, optimizer=adam, metrics=['accuracy'])
model.compile(loss=dice_loss, optimizer=adam, metrics=['accuracy'])
model.summary()
weight = "Model/DRIVE/DRIVE.h5"
restore = True

if restore and os.path.isfile(weight):
    model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=False)

def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)


history = model.fit(x_train, y_train,
                epochs=150,
                batch_size=8,
                validation_data=(x_validate, y_validate),
                shuffle=True,
                callbacks=[TensorBoard(log_dir='./autoencoder'), reduce_lr, model_checkpoint], verbose=2)


print(history.history.keys())

# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('SA-UNet Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='lower right')
# plt.show()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('SA-UNet Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper right')
# plt.show()

