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


#
#
#
#
#
RUN = "6"
mode = "3d"
batch_size = 64 # # 64 # 128
nb_classes = 2
nb_epoch = 400
lr = 0.000001 # 0.000001
# with 108 
# 0.000001 works with 192 last layer but resource exhausted
# works : 0.0000001
# no list
# 0.01 - 0.1 - 1.0 - 5.0
count = 2 # for 3d mode, no i=og images to take in every direction
imgSize = 64 # 108
#
augmentationFactor =   1 #9*3 # original + 3 flips + 2 rotations
augment3d = False
#
#
#
#
#
funcs.RUN = RUN
funcs.mode = mode
funcs.imgSize = imgSize
funcs.count = count
funcs.augmentationFactor = augmentationFactor


dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames()
#
x_train_a , x_train_s , x_train_c , y_train , zeros , ones , clinical = funcs.getXandY(dataFrameTrain, mode, imgSize, count, False)
print ("train data: (will not match zof of augemntation later) " ,x_train_a.shape , x_train_s.shape  , x_train_c.shape  , y_train.shape , clinical.shape ) 

print ("zeros: " , zeros , "ones: " , ones)
zeroWeight = ones / ((ones+zeros)*1.0)
oneWeight = zeros / ((ones+zeros)*1.0)
print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)

with tf.device('/gpu:0'):

    # keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', histogram_freq=0, write_graph=True, write_images=False)

    histories = funcs.Histories()

    model_0 = funcs.makeClinicalModel()

    if mode == "3d":
        model_A = funcs.make3dConvModel(imgSize,count)  
        model_S = funcs.make3dConvModel(imgSize,count)  
        model_C = funcs.make3dConvModel(imgSize,count) 
    elif mode == "2d":
        model_A = funcs.make2dConvModel(imgSize)  
        model_S = funcs.make2dConvModel(imgSize)  
        model_C = funcs.make2dConvModel(imgSize)     

    model = Sequential()
    # 
    model.add(keras.engine.topology.Merge([  model_0 , model_A, model_S, model_C  ], mode='concat', concat_axis=1)) # 512*3 + 3

    model.add(Dense(512))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # myOptimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    myOptimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # myOptimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0) 
    # myOptimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #
    model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])

    # save model
    plot(model, show_shapes=True , show_layer_names=True,  to_file='/home/ubuntu/output/' + RUN + '_model.png')


    # center and standardize all including augmented
    mean_a,std_a,x_train_a = funcs.centerAndStandardizeTraining(x_train_a)
    mean_s,std_s,x_train_s = funcs.centerAndStandardizeTraining(x_train_s)
    mean_c,std_c,x_train_c = funcs.centerAndStandardizeTraining(x_train_c)

    # pass back to funcs
    funcs.mean_a = mean_a
    funcs.std_a = std_a
    funcs.mean_s = mean_s
    funcs.std_s = std_s
    funcs.mean_c = mean_c
    funcs.std_c = std_c


    print ( "params: " , model.count_params() )

    print ( "samples_per_epoch: " , ( x_train_a.shape[0] - (x_train_a.shape[0]%batch_size) ) )


    if mode == "2d":

        datagenTrain = ImageDataGenerator(
            #
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            #
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.4,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.4,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True, # randomly flip images
            # #
            # shear_range=0.,
            # zoom_range=0.,
            # channel_shift_range=0.,
            # #
            fill_mode='reflect',
            # cval=0.,
            # rescale=None,
            # dim_ordering=K.image_dim_ordering()
            )  



        print ("2d mode training final shapes " , x_train_a.shape , x_train_s.shape , x_train_c.shape , y_train.shape , clinical.shape )

        model.fit_generator( funcs.createGenerator(clinical, x_train_a,x_train_s,x_train_c,y_train,batch_size,datagenTrain) ,
                            samples_per_epoch= ( x_train_a.shape[0] - (x_train_a.shape[0]%batch_size) ) ,
                            class_weight={0 : zeroWeight, 1: oneWeight},
                            nb_epoch=nb_epoch,
                           callbacks=[histories])


    elif mode == "3d":

        # do augmentation first, adds more data
        # only does flipping , add rotation and shifting, contrast, brightness..
        if (augment3d):
            x_train_a , x_train_s , x_train_c  = funcs.augmentTraining(x_train_a , x_train_s , x_train_c,mode)

        print ("3d mode training final shapes " , x_train_a.shape , x_train_s.shape , x_train_c.shape , y_train.shape , clinical.shape )


        # NO GENERATOR
        model.fit ( [ clinical, x_train_a , x_train_s , x_train_c ] , y_train  , 
            batch_size=batch_size, 
            nb_epoch=nb_epoch, 
            verbose=1, 
            callbacks=[histories], 
            validation_split=0.0, 
            validation_data=None, 
            shuffle=True, 
            class_weight={0 : zeroWeight, 1: oneWeight}, 
            sample_weight=None, 
            initial_epoch=0)