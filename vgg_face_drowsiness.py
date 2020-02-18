import os
import cv2
import numpy as np
import keras.backend as K
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.initializers import he_normal
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Input, Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, concatenate, GlobalMaxPooling2D

input_face = Input(shape = (75,75,3))

x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', kernel_initializer = 'he_normal', 
	trainable = False)(input_face)
x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2', kernel_initializer = 'he_normal', 
	trainable = False)(x)
x = MaxPooling2D(pool_size=(2,2), name='block1_pool', trainable = False)(x)

x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1', kernel_initializer = 'he_normal', 
	trainable = False)(x)
x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2', kernel_initializer = 'he_normal', 
	trainable = False)(x)
x = MaxPooling2D(pool_size=(2,2), name='block2_pool', trainable = False)(x)

x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1', kernel_initializer = 'he_normal', 
	trainable = False)(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2', kernel_initializer = 'he_normal', 
	trainable = False)(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3', kernel_initializer = 'he_normal', 
	trainable = False)(x)
x1 = MaxPooling2D(pool_size=(2,2), name='block3_pool', trainable = False)(x)

x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', name='block4_conv1', )(x1)
x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', name='block4_conv2', )(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', name='block4_conv3', )(x)
x2 = MaxPooling2D(pool_size=(2,2), name='block4_pool')(x)

x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', name='block5_conv1', )(x2)
x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', name='block5_conv2', )(x)
x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer = 'he_normal', name='block5_conv3', )(x)
x3 = MaxPooling2D(pool_size=(2,2), name='block5_pool')(x)

vgg16_model = Model(input_face, x3)

vgg16_model.load_weights('/home/vision/Desktop/HFDD/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

convx1a = Conv2D(256, (1,1), activation="relu", padding="same", kernel_initializer = 'he_normal', 
	kernel_regularizer=regularizers.l2(0.001))(vgg16_model.get_layer('block3_pool').output)
poolx1a = GlobalMaxPooling2D()(convx1a)

convx1b = Conv2D(256, (1,1), activation="relu", padding="same", kernel_initializer = 'he_normal', 
	kernel_regularizer=regularizers.l2(0.001))(vgg16_model.get_layer('block4_pool').output)
poolx1b = GlobalMaxPooling2D()(convx1b)

convx1c = Conv2D(256, (1,1), activation="relu", padding="same", kernel_initializer = 'he_normal', 
	kernel_regularizer=regularizers.l2(0.001))(vgg16_model.get_layer('block5_pool').output)
poolx1c = GlobalMaxPooling2D()(convx1c)

merged = concatenate([poolx1a, poolx1b, poolx1c])
batch_norm3 = BatchNormalization()(merged)
fc1 = Dense(768, activation="relu", kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.001))(batch_norm3)
dr1 = Dropout(0.5)(fc1)

batch_norm4 = BatchNormalization()(dr1)
fc2	= Dense(256, activation="relu", kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.001))(batch_norm4)
dr2 = Dropout(0.25)(fc2)

batch_norm5 = BatchNormalization()(dr2)
fc3 = Dense(2, activation="softmax")(batch_norm5)

final_model = Model(input_face, fc3)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0001, amsgrad=False)
final_model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])

print(final_model.summary())

#######################################################################################################################################

batch_size = 32
epochs = 200

print("\n \t [INFO] Loading the Face Data from Training Set and Evaluation Set of the Dataset. \n")

x_train = np.load('/media/vision/Data/face_data_75_npy/xtrain.npy')
y_train = np.load('/media/vision/Data/face_data_75_npy/ytrain.npy')
x_val = np.load('/media/vision/Data/face_data_75_npy/xval.npy')
y_val = np.load('/media/vision/Data/face_data_75_npy/yval.npy')

tb = TensorBoard(log_dir='./tb_dir_vgg16_new', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True, 
	embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

filepath_for_best_val_acc = "/media/vision/Data/vgg16_new_cnn_adam_point001_face_3ch_75_best_val_acc_bs32.hdf5"
checkpoint = ModelCheckpoint(filepath_for_best_val_acc, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
earlystopping = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 25, verbose = 0, mode = 'auto', baseline = None, 
	restore_best_weights = False)
callbacks_list = [checkpoint, tb]

print("\n [INFO] The Convolutional Neural Network is starting to learn... \n")

history = final_model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs, callbacks = [checkpoint, tb], 
	verbose = 1, shuffle = True, validation_data = (x_val, y_val))

model.save('/media/vision/Data/vgg16_new_cnn_adam_point001_face_3ch_75_final_model_bs32.hdf5')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('CNN Model Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.savefig('/media/vision/Data/vgg16_new_cnn_adam_point001_face_3ch_75_bs32_acc.png')
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Model Loss Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper right')
plt.savefig('/media/vision/Data/vgg16_new_cnn_adam_point001_face_3ch_75_bs32_loss.png')
plt.close()

print(" [INFO] Training Completed! Accuracies and Loss Values achieved are displayed with each Epoch.\
	Saving the best_val_acc model, final model, and graphs to disk...")