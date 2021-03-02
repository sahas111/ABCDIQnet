__author__= 'Susmita Saha'

'''This 3DCNN model predicts actual fluid intelligence scores from the 60-60-3 patches out of the slice input with the best predictive power'''

import numpy as np
import tensorflow as tf
import sys
import Functionlib
import h5py
import scipy.stats
import pandas as pd

sys.path.append('/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/ABCDIQNet/')

#Define the command-line arguments
slice_range_start=int(sys.argv[1])
slice_range_end=int(sys.argv[2])
print ('slice_range_start',slice_range_start)
print ('slice_range_end',slice_range_end)

#create a dataframe for storing results
df = pd.DataFrame()
df.columns = ['slice_start_no','slice_end_no','val_MSE','val_corr','train_MSE','train_corr']
#Read the train data
h5f = h5py.File('/scratch1/sah012/ABCD_T1/patch_train_data/' + 'ABCD_T1_IQ_train_patch_60_60_3_'+str(slice_range_start)+'_'+str(slice_range_end)+'.h5', 'r')
X_train = h5f['x_T1_train'][:]
Y_train_regress = h5f['y_train_FIQ_trainScore_regression'][:]
h5f.close()

#Read the validation data
h5f = h5py.File('/scratch1/sah012/ABCD_T1/patch_val_data/' + 'ABCD_T1_IQ_val_patch_60_60_3_'+str(slice_range_start)+'_'+str(slice_range_end)+'.h5', 'r')
X_val = h5f['x_T1_val'][:]
Y_val_regress = h5f['y_val_FIQ_valScore_regression'][:]
h5f.close()

#Prepare the data with patches as channel inputs
X_train = np.asarray(X_train)
Y_train_regress = np.asarray(Y_train_regress)
X_train = X_train.transpose(0, 2, 3, 4, 1)
Shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])

X_val = np.asarray(X_val)
Y_val_regress = np.asarray(Y_val_regress)
X_val = X_val.transpose(0, 2, 3, 4, 1)

#CNN architecture
weight_decay=0.00005
kernel_initializer='glorot_uniform'
reg='l2'
if reg == 'l1':
    reg_func = tf.keras.regularizers.l1
else:
    reg_func = tf.keras.regularizers.l2

input_layer = tf.keras.layers.Input(shape=Shape)
x =tf.keras.layers.Conv3D(8, (3, 3, 1), padding='valid', kernel_initializer=kernel_initializer, name='conv1',
           kernel_regularizer=reg_func(weight_decay))(input_layer)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv3D(8, (3, 3, 1), padding='valid', kernel_initializer=kernel_initializer, name='conv2',
           kernel_regularizer=reg_func(weight_decay))(x)
x = tf.keras.layers.BatchNormalization(momentum=0.99)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1))(x)
x =tf.keras.layers.Conv3D(16, (3, 3, 1), padding='valid', kernel_initializer=kernel_initializer, name='conv3',
           kernel_regularizer=reg_func(weight_decay))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv3D(16, (3, 3, 1), padding='valid', kernel_initializer=kernel_initializer, name='conv4',
           kernel_regularizer=reg_func(weight_decay))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1))(x)
x = tf.keras.layers.Conv3D(32, (3, 3, 1), padding='valid', kernel_initializer=kernel_initializer, name='conv5',
           kernel_regularizer=reg_func(weight_decay))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv3D(32, (3, 3, 1), padding='valid', kernel_initializer=kernel_initializer, name='conv6',
           kernel_regularizer=reg_func(weight_decay))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1))(x)
x = tf.keras.layers.Flatten()(x)
x =tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
predictions = tf.keras.layers.Dense(1, name='fcdense', kernel_initializer='he_normal')(x)
model=tf.keras.models.Model(inputs=input_layer, outputs=predictions)
print ('model_summary', model.summary())

#Define the model paramters and hyperparameters
NbEpochs =50
BatchSize = 64
LearningRate=0.0001
Do2D=True
TrainModel = True
TotalTrainingSamples=len(X_train)
TotalValidationSamples=len(X_val)

#Define generators
training_generator = Functionlib.data_generator(X_train,Y_train_regress,Shape,BatchSize)
validation_generator = Functionlib.data_generator(X_val,Y_val_regress,Shape,BatchSize)

#Define the path to logs and saved models
PthModel='/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/models_patch/model_60_60_3_2.hdf5'

# Compile the CNN
adam = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

# Train a new model
if TrainModel:
    # Define callback to ensure we save the best performing model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=PthModel, monitor='val_loss', verbose=2, save_best_only=True)
    cb = [checkpointer]
    model.fit_generator(training_generator,
                        epochs=NbEpochs,
                        validation_data=validation_generator,
                        steps_per_epoch=TotalTrainingSamples/BatchSize,
                        validation_steps=TotalValidationSamples/BatchSize,
                        callbacks=cb,verbose=2)
    del model
    model = tf.keras.models.load_model(PthModel)

# validation results
yp = model.predict(X_val)
val_mse = np.mean(np.square(np.array(Y_val_regress) - yp))
Y_val_regress = Y_val_regress.reshape(Y_val_regress.shape[0], 1)
Y_val_regress = np.concatenate(Y_val_regress)
yp = np.concatenate(yp)
val_correlation_pearson = scipy.stats.pearsonr(yp, Y_val_regress)
# training results
yt = model.predict(X_train)
train_mse = np.mean(np.square(np.array(Y_train_regress) - yt))
Y_train_regress = Y_train_regress.reshape(Y_train_regress.shape[0], 1)
Y_train_regress = np.concatenate(Y_train_regress)
yt = np.concatenate(yt)
train_correlation_pearson = scipy.stats.pearsonr(yt, Y_train_regress)
# performance results to pandas dataframe
df['slice_start_no'].append(slice_range_start)
df['slice_end_no'].append(slice_range_end)
df['val_MSE'].append(val_mse)
df['val_corr'].append(val_correlation_pearson)
df['train_MSE'].append(train_mse)
df['train_corr'].append(train_correlation_pearson)

# Save final results as a csv file
df.to_csv(
r'/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/ABCDIQNet/results_train_val_patchmodel_2.csv',
header=True)
