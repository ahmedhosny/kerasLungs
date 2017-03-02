
from __future__ import division
import funcs
import numpy as np
import keras
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

print('March1-1')

dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames()
#
x_train_a , x_train_s , x_train_c , y_train , zeros , ones = funcs.getXandY(dataFrameTrain)
print ("train data: " ,x_train_a.shape , x_train_s.shape  , x_train_c.shape  , y_train.shape )
print ("zeros: " , zeros , "ones: " , ones)
zeroWeight = ones / ((ones+zeros)*1.0)
oneWeight = zeros / ((ones+zeros)*1.0)
print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)

with tf.device('/gpu:0'):

    # keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', histogram_freq=0, write_graph=True, write_images=False)

    batch_size = 64 # 128
    nb_classes = 2
    nb_epoch = 400


    histories = funcs.Histories()

    model_A = funcs.make2dConvModel()  #########################################
    model_S = funcs.make2dConvModel()  #########################################
    model_C = funcs.make2dConvModel()  #########################################

    model = Sequential()
    model.add(keras.engine.topology.Merge([ model_A, model_S, model_C  ], mode='concat', concat_axis=1))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    myOptimizer = keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
    # myOptimizer = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0) 
    model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])

    # save model
    plot(model, show_shapes=True , show_layer_names=True,  to_file='/home/ubuntu/output/' + funcs.RUN + '_model.png')

    datagenTrain = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.4,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.4,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


    datagenTrain.fit(x_train_a)
    datagenTrain.fit(x_train_s)
    datagenTrain.fit(x_train_c)

    print ( "samples_per_epoch: " , ( x_train_a.shape[0] - (x_train_a.shape[0]%batch_size) ) )


    model.fit_generator( funcs.createGenerator(x_train_a,x_train_s,x_train_c,y_train,batch_size,datagenTrain) ,
                        samples_per_epoch= ( x_train_a.shape[0] - (x_train_a.shape[0]%batch_size) ) ,
                        class_weight={0 : zeroWeight, 1: oneWeight},
                        nb_epoch=nb_epoch,
                       callbacks=[histories])

