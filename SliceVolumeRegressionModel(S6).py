__author__= 'Susmita Saha'

'''This model is to predict actual fluid intelligence scores from the best perfroming slice input and brain regional volume measures'''

import numpy as np
import tensorflow as tf
import sys
import Functionlib
import h5py
import scipy.stats
sys.path.append('/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/ABCDIQNet/')

#Normalise both dataset
def normalise_z(array_np):
    # Normalise NumPy array (zero mean, unit variance)
    array_np_norm = (array_np - np.mean(array_np)) / np.std(array_np)
    return array_np_norm

#Read Train data
h5f = h5py.File('/scratch1/sah012/ABCD_T1/slice_train_data/' + 'ABCD_T1_IQ_train3Sliceactual_LR_thalamus_hippo_FIQ_96_98.h5', 'r')
X_train_actual=h5f['x_T1_train_actual'][:]
Y_train_regress_actual=h5f['y_train_actual_FIQ_trainScore_regression'][:]
left_hippocampus_GM_vol_train=h5f['left_hippocampus_GM_vol_train'][:]
right_hippocampus_GM_vol_train=h5f['right_hippocampus_GM_vol_train'][:]
left_thalamus_GM_vol_train=h5f['left_thalamus_GM_vol_train'][:]
right_thalamus_GM_vol_train=h5f['right_thalamus_GM_vol_train'][:]
h5f.close()

X_train_actual=normalise_z(X_train_actual)
Y_train_regress = np.array(Y_train_regress_actual,dtype=float)
left_thalamus_GM_vol_train = np.array(left_thalamus_GM_vol_train,dtype=float)
left_thalamus_GM_vol_train=normalise_z(left_thalamus_GM_vol_train)
left_thalamus_GM_vol_train = left_thalamus_GM_vol_train.reshape(left_thalamus_GM_vol_train.shape[0],1)
print ('score_data_shape:{0}').format(Y_train_regress.shape)
print ('volume_data_shape:{0}').format(left_thalamus_GM_vol_train.shape)

#Read Val data
h5f = h5py.File('/scratch1/sah012/ABCD_T1/slice_val_data/' + 'ABCD_T1_IQ_val3Sliceactual_LR_thalamus_hippo_FIQ_96_98.h5', 'r')
X_val_actual = h5f['x_T1_val_actual'][:]
Y_val_regress_actual=h5f['actual_FIQ_val_label_regression'][:]
left_thalamus_GM_vol_val=h5f['left_thalamus_GM_vol_val'][:]
h5f.keys()
h5f.close()

X_val_actual=normalise_z(X_val_actual)
Y_val_regress = np.array(Y_val_regress_actual,dtype=float)
left_thalamus_GM_vol_val  = np.array(left_thalamus_GM_vol_val,dtype=float)
left_thalamus_GM_vol_val=normalise_z(left_thalamus_GM_vol_val)
left_thalamus_GM_vol_val=left_thalamus_GM_vol_val.reshape(left_thalamus_GM_vol_val.shape[0],1)

#Define the model parameters and hyperparameters
TotalTrainingSamples=len(X_train_actual)
TotalValidationSamples=len(X_val_actual)
Shape1 = (X_train_actual[0].shape[0],X_train_actual[0].shape[1],3)
Shape2 = (1,)
TrainModel = True
weight_decay=0.00005
kernel_initializer='glorot_uniform'
reg='l2'
if reg == 'l1':
    reg_func = tf.keras.regularizers.l1
else:
    reg_func = tf.keras.regularizers.l2

NbEpochs = 50
BatchSize = 64
LearningRate = 1e-4

#Define the path to save and restore trained models
DirLog='/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/log/'
PthModel='/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/models_seg/model.hdf5'

#Define a custom layer for concatenation of data in Keras if required
def concat_layer(tensor1):
    return tf.concat(tensor1,axis=-1)

#CNN architecture
input_layer1 = tf.keras.layers.Input(shape=Shape1)
x =tf.keras.layers.Conv2D(32, 3, padding="same",kernel_initializer='he_normal')(input_layer1)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x =tf.keras.layers.Conv2D(32, 3, padding="same",kernel_initializer='he_normal' )(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x =tf.keras.layers.Conv2D(16, 3, padding="same",kernel_initializer='he_normal' )(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x =tf.keras.layers.Conv2D(16, 3, padding="same",kernel_initializer='he_normal' )(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x =tf.keras.layers.Conv2D(8, 3, padding="same",kernel_initializer='he_normal' )(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x =tf.keras.layers.Conv2D(8, 3, padding="same",kernel_initializer='he_normal' )(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x =tf.keras.layers.Dense(1024, activation='relu')(x)
x =tf.keras.layers.Dense(256, activation='relu')(x)
model_cnn=tf.keras.models.Model(inputs=input_layer1, outputs=x)

#multi-layer-perceptron architecture
input_layer2 = tf.keras.layers.Input(shape=Shape2)
x =tf.keras.layers.Dense(1024, activation='relu')(input_layer2)
x =tf.keras.layers.Dense(512, input_dim=Shape2, activation="relu")(x)
x =tf.keras.layers.Dense(256, input_dim=Shape2, activation="relu")(x)
model_mlp=tf.keras.Model(inputs=input_layer2,outputs=x)

#concatenate two model outputs
combinedInput = tf.keras.layers.concatenate([model_mlp.output, model_cnn.output])
#Add a final FC layer for the combined input
x = tf.layers.Dense(256, activation="relu")(combinedInput)
x = tf.layers.Dense(1, activation="linear")(x)

Combined_model = tf.keras.Model(inputs=[input_layer1,input_layer2], outputs=x)
Combined_model.summary()
#Compile CNN model
adam = tf.keras.optimizers.Adam(lr = LearningRate)
Combined_model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
#Define generators
training_generator = Functionlib.data_generator_multiple(X_train_actual,left_thalamus_GM_vol_train,Y_train_regress,Shape1,Shape2,BatchSize)
validation_generator = Functionlib.data_generator_multiple(X_val_actual,left_thalamus_GM_vol_val,Y_val_regress,Shape1,Shape2,BatchSize)

#Train a new model
if TrainModel:

    # Define callback to ensure we save the best performing model
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=PthModel, monitor='val_loss', verbose=0, save_best_only=True)
    cb = [checkpointer]
    Combined_model.fit_generator(training_generator,
                        epochs=NbEpochs,
                        validation_data=validation_generator,
                        steps_per_epoch=TotalTrainingSamples/BatchSize,
                        validation_steps=TotalValidationSamples/BatchSize,
                        callbacks=cb,verbose=2)
    del Combined_model
    Combined_model = tf.keras.models.load_model(PthModel)
print ('predict results')
#validation
yp = Combined_model.predict([X_val_actual,left_thalamus_GM_vol_val])
mse = np.mean(np.square(np.array(Y_val_regress) - yp))
print('val MSE = ' + str(mse))
Y_val_regress=Y_val_regress.reshape(Y_val_regress.shape[0],1)
Y_val_regress=np.concatenate(Y_val_regress)
yp=np.concatenate(yp)

print ('val_predictions',yp)
print ('val_actuals',Y_val_regress)
print ('predictions_shape',yp.shape)
print ('actual_shape',Y_val_regress.shape)
print ('val correlation',np.corrcoef(yp, Y_val_regress)[0,1])
print ('val_correlation',np.corrcoef(yp, Y_val_regress)[1,0])
print ('val correlation',scipy.stats.pearsonr(yp, Y_val_regress))

#training
yt = Combined_model.predict([X_train_actual,left_thalamus_GM_vol_train])
train_mse = np.mean(np.square(np.array(Y_train_regress) - yt))
print('Train MSE = ' + str(train_mse))
print('val MSE = ' + str(mse))
Y_train_regress=Y_train_regress.reshape(Y_train_regress.shape[0],1)
Y_train_regress=np.concatenate(Y_train_regress)
yt=np.concatenate(yt)
print ('actual_shape',Y_train_regress.shape)
print ('train correlation',np.corrcoef(yt, Y_train_regress)[1,0])
print ('train correlation',scipy.stats.pearsonr(yt, Y_train_regress))

