# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from sklearn.preprocessing import QuantileTransformer
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import numpy as np
import math
import os
import random
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from tensorflow.python.framework import ops

# Global Settings
batch_size= 2
num_epochs = 1000
learning_rate = 1e-4
img_size = 2048
img_channels=1
for i in range(1,31):
    file_name = f'x{i}'
    faults.append(file_name)

root = './LYan-SEDEP/results_ddpm/data_pu/gen_'
# root = './LYan-SEDEP/results_ddpm/data_chopper/gen_'
for fault in faults:
    # Build graph
    ops.reset_default_graph()
    # Build encoder
    inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
    # 2，神经网络
    layers = tf.keras.layers
    # 128
    conv1 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
    maxpool1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
    maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
    maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv4 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool3)
    maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
    conv5 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool4)

    maxpool5 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv5)
    re = tf.reshape(maxpool5, [-1, 128])
    # -----------#
    latent = layers.Dense(units=128)(re)
    # latent = layers.Dense(units=128, activation=tf.nn.relu)(re)
    # -----------#
    # ---Decoder---#
    x = layers.Dense(units=128, activation=tf.nn.relu)(re)
    x = tf.reshape(x, [-1, 64, 2])
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    print(rx.shape, inputs_.shape)
    print('Built Encoder../')
    # print(image_input.shape, enout.shape, x_out.shape)
    # #Build model
    dcae=keras.Model(inputs_, rx)
    #
    # # Opimizer and loss function
    opt = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
    print('Network Summary-->')
    # dcae.summary()
    save_dir = './LYan-SEDEP/DCAE/model_pu/model_last_999_128.ckpt'
    # save_dir = './LYan-SEDEP/DCAE/model_chopper/model_last_999_128.ckpt'
    print('Load weights from ', dir)
    dcae.load_weights(dir)
    new_enout=tf.keras.models.Model(inputs=inputs_,outputs=latent)

    #data_test
    file_name = root + fault + '.pkl'
    x_test = pickle.load(open(file_name, 'rb'))[1]
    data_test = tf.reshape(x_test, shape=[-1, 2048, 1])
    extracted_features_test = new_enout.predict(data_test)
    print(extracted_features_test.shape)
    with open('./LYan-SEDEP/DCAE/results_pu/en_gen_' + fault + '_test.pkl', 'wb') as f:
    # with open('./LYan-SEDEP/DCAE/results_chopper/en_gen_' + fault + '_test.pkl', 'wb') as f:
        pickle.dump(extracted_features_test, f, pickle.HIGHEST_PROTOCOL)

    #data_train
    x_train = pickle.load(open(file_name, 'rb'))[0]
    data_train = tf.reshape(x_train, shape=[-1, 2048, 1])
    extracted_features_train = new_enout.predict(data_train)
    print(extracted_features_train.shape)
    with open('./LYan-SEDEP/DCAE/results_pu/en_gen_' + fault + '_train.pkl', 'wb') as f:
    # with open('./LYan-SEDEP/DCAE/results_chopper/en_gen_' + fault + '_train.pkl', 'wb') as f:
        pickle.dump(extracted_features_train, f, pickle.HIGHEST_PROTOCOL)





