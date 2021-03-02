__author__= 'Susmita Saha'

'''This file contains all the necessary functions for data preparation and model configuration'''

from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding3D, Conv3D, Activation, MaxPooling3D, Average, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import random
from tensorflow.keras.layers import BatchNormalization
import numpy as np

def data_generator(X, y, Shape, BatchSize=8):
    # Returns a Tensorflow Keras generator.
    # X     - Inputs
    # y     - A list of regression targets
    while True:
        batch_idx = 0
        shuffled_index = list(range(len(X)))
        random.shuffle(shuffled_index)
        for i in shuffled_index:
            x1 = np.zeros((BatchSize,) + Shape, dtype=np.float32)
            y1 = np.zeros((BatchSize, 1), dtype=np.float32)
            x1[batch_idx % BatchSize] = X[i]
            y1[batch_idx % BatchSize] = y[i]
            batch_idx += 1
            if (batch_idx % BatchSize) == 0:
                yield (x1, y1)

def get_CNN_model2(Shape, filters=([64, 2], [128, 2], [256, 2]), fully_connected=([500, 0.5], [100, 0.25], [20, 0]),
                  batchNorm=True):
    # Returns a Keras regression CNN model.
    inputShape = Shape
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        if i == 0:
            x = inputs
            # CONV => RELU => BN => POOL
        for i1 in range(0, f[1]):
                x = Conv2D(f[0], 3, padding="same",kernel_initializer='he_normal' )(x)
            x = Activation("relu")(x)
        if batchNorm:
            x = BatchNormalization(axis=chanDim)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the volume...
    x = Flatten()(x)
    if isinstance(fully_connected, tuple):
        for (i, f) in enumerate(fully_connected):
            x = Dense(f[0])(x)
            x = Activation("relu")(x)
            if batchNorm:
                x = BatchNormalization(axis=chanDim)(x)
            if f[1] > 0:
                x = Dropout(f[1])(x)
    else:
        x = Dense(fully_connected[0])(x)
        x = Activation("relu")(x)
        if batchNorm:
            x = BatchNormalization(axis=chanDim)(x)
        if fully_connected[1] > 0:
            x = Dropout(fully_connected[1])(x)
            # Add the regression node
    x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
    # return the CNN
    return model
def data_generator_multiple(X1, X2, y, Shape_1,Shape_2, BatchSize=8, Do2D=False):
    # Returns a Tensorflow Keras generator.
    # X     - Inputs
    # y     - A list of regression targets
    # Shape - Size of a training sample
    # ___________________________________________________________________

    while True:
        batch_idx = 0
        shuffled_index = list(range(len(X1)))
        random.shuffle(shuffled_index)

        for i in shuffled_index:
            x1 = np.zeros((BatchSize,) + Shape_1, dtype=np.float32)
            x2 = np.zeros((BatchSize,) + Shape_2, dtype=np.float32)
            y1 = np.zeros((BatchSize, 1), dtype=np.float32)

            x1[batch_idx % BatchSize] = X1[i]
            x2[batch_idx % BatchSize] = X2[i]
            y1[batch_idx % BatchSize] = y[i]
            batch_idx += 1

            if (batch_idx % BatchSize) == 0:
                yield ([x1,x2], y1)
