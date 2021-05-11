import os
import scipy.misc as mc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from keras import losses
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *

data_location = './'
training_images_loc = data_location + 'CHASE/train/image/'
training_label_loc = data_location + 'CHASE/train/label/'
validate_images_loc = data_location + 'CHASE/validate/images/'
validate_label_loc = data_location + 'CHASE/validate/labels/'
train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []
desired_size = 1008

for i in train_files:
    im = mc.imread(training_images_loc + i)
    label = mc.imread(training_label_loc + i.split('_')[0]+"_"+i.split('_')[1].split(".")[0] +"_1stHO.png" ,mode="L")
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

    temp = cv2.resize(new_label,
                      (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)



for i in validate_files:
    im = mc.imread(validate_images_loc + i)
    label = mc.imread(validate_label_loc +i.split('_')[0]+'_'+ i.split('_')[1].split(".")[0] +"_1stHO.png" ,mode="L")
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

    temp = cv2.resize(new_label,
                      (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    validate_label.append(temp)
train_data = np.array(train_data)

train_label = np.array(train_label)
validate_data = np.array(validate_data)
validate_label = np.array(validate_label)




x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

x_validate = validate_data.astype('float32') / 255.
y_validate = validate_label.astype('float32') / 255.
x_validate = np.reshape(x_validate, (len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_validate = np.reshape(y_validate, (len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im


TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

from UNet import Ghost_UNet
from UNet import ESA_UNet
model = ESA_UNet(input_size=(desired_size, desired_size, 3), start_neurons=16, lr=1e-3, keep_prob=0.87, block_size=7)
adam = Adam(lr=0.001, epsilon=1e-8)

def dice_loss(target, pred):
    """dice loss function for tensorflow/keras
        calculate dice loss per batch and channel of each sample.
    Args:
        data_format: either channels_first or channels_last
    Returns:
        loss_function(y_true, y_pred)
    """
    if K.image_data_format() == "channels_last":
        pred = tf.transpose(pred, (0, 3, 1, 2))
        target = tf.transpose(target, (0, 3, 1, 2))

    smooth = 1.0
    iflat = tf.reshape(
        pred, (tf.shape(pred)[0], tf.shape(pred)[1], -1)
    )  # batch, channel, -1
    tflat = tf.reshape(target, (tf.shape(target)[0], tf.shape(target)[1], -1))
    intersection = K.sum(iflat * tflat, axis=-1)
    return 1 - ((2.0 * intersection + smooth)) / (
            K.sum(iflat, axis=-1) + K.sum(tflat, axis=-1) + smooth
    )

model.compile(loss=losses.binary_crossentropy, optimizer=adam, metrics=['accuracy'])
model.summary()
weight = "Model/CHASE/Ghost_CHASE.h5"
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
                epochs=300,
                batch_size=4,
                validation_data=(x_validate, y_validate),
                shuffle=True,
                callbacks=[TensorBoard(log_dir='./autoencoder'), reduce_lr, model_checkpoint], verbose=2)


print(history.history.keys())

# summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('SA-UNet Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='lower right')
# plt.show()
#
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('SA-UNet Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validate'], loc='upper right')
# plt.show()