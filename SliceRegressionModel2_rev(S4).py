__author__= 'Susmita Saha'

"""This 2DCNN model (additional) with reverse configuration is used for residualised score prediction from adjacent three axial slices"""

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
df = pd.DataFrame(columns=np.arange(6))
df.columns = ['slice_start_no','slice_end_no','val_MSE','val_corr','train_MSE','train_corr']
i=0
for slice_start_no in range(slice_range_start, slice_range_end, 3):
    slice_start=int(slice_start_no)
    slice_end=int(slice_start_no)+2
    h5f = h5py.File('/scratch1/sah012/ABCD_T1/slice_train_data/' + 'ABCD_T1_IQ_train3Sliceactual_'+str(slice_start)+'_'+str(slice_end)+'.h5', 'r')
    X_train = h5f['x_T1_train_actual'][:]
    Y_train_regress_actual=h5f['y_train_actual_FIQ_train_uncorrectedScore_regression'][:]
    h5f.close()
    Y_train_regress = np.array(Y_train_regress_actual,dtype=float)
    h5f = h5py.File('/scratch1/sah012/ABCD_T1/slice_val_data/' + 'ABCD_T1_IQ_val3Sliceactual_'+str(slice_start)+'_'+str(slice_end)+'.h5', 'r')
    X_val = h5f['x_T1_val_actual'][:]
    Y_val_regress_actual=h5f['y_val_actual_FIQ_val_uncorrectedScore_regression'][:]
    h5f.close()
    Y_val_regress = np.array(Y_val_regress_actual,dtype=float)
    #Define the path to save models and logs
    PthModel='/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/models2_rev/model_2_rev_'+str(slice_start)+'_'+str(slice_end)+'.hdf5'
    #Define the model parameters and hyperparameters
    Shape = (X_train[0].shape[0],X_train[0].shape[1],3)
    Model2D = True
    TrainModel = True
    TotalTrainingSamples=len(X_train)
    TotalValidationSamples=len(X_val)
    NbEpochs = 50
    BatchSize = 16
    LearningRate = 1e-4
    #Define generators
    training_generator = Functionlib.data_generator(X_train,Y_train_regress,Shape,BatchSize,Do2D)
    validation_generator = Functionlib.data_generator(X_val,Y_val_regress,Shape,BatchSize,Do2D)
    model = Functionlib.get_CNN_model2(Shape,
                              Model2D=,
                              filters = ([32,1],[32, 1],[16,1],[16, 1],[8,1],[8,1]),
                              fully_connected = ([1024,0],[256,0]),
                              batchNorm = False)

    model.summary()
    # Compile the CNN
    adam = tf.keras.optimizers.Adam(lr=LearningRate)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
    # Train a new model
    if TrainModel:
        # Define callback to ensure we save the best performing model
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=PthModel, monitor='val_loss', verbose=0,
                                                          save_best_only=True)

        cb = [checkpointer]
        model.fit_generator(training_generator,
                            epochs=NbEpochs,
                            validation_data=validation_generator,
                            steps_per_epoch=TotalTrainingSamples / BatchSize,
                            validation_steps=TotalValidationSamples / BatchSize,
                            callbacks=cb, verbose=2)
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
    # performance results to pandas dataframe to csv
    df.loc[i,'slice_start_no']=slice_start
    df.loc[i,'slice_end_no']=slice_end
    df.loc[i,'val_MSE']=val_mse
    df.loc[i,'val_corr']=val_correlation_pearson
    df.loc[i,'train_MSE']=train_mse
    df.loc[i,'train_corr']=train_correlation_pearson

    i+=1

# Save final results as a csv file
df.to_csv(
    r'/datastore/sah012/CNN-3D-images-Tensorflow/super-resolution-project/ABCD-models/ABCDIQNet/results_train_val_slicemodel_2.csv',
    header=True)









