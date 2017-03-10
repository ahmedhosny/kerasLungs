from __future__ import division
import funcs
import krs
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
RUN = "10"
mode = "3d"
batch_size = 64 # # 64 # 128
nb_classes = 2
nb_epoch = 1000
lr = 0.000001 # 0.000001
count = 5 # for 3d mode, no i=og images to take in every direction
finalSize = 120 # from 150 down to..
imgSize = 100 # 108
valTestMultiplier = 1

# for single3d
fork = False
skip = 2 # (imgSize/skip should be int)

#
#
#
#
#
#
funcs.RUN = RUN
funcs.mode = mode
funcs.imgSize = imgSize
funcs.count = count
funcs.valTestMultiplier = valTestMultiplier
funcs.finalSize = finalSize
funcs.fork = fork
funcs.skip = skip


dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames()
#
x_train,y_train,zeros,ones,clinical_train =  funcs.getXandY(dataFrameTrain,imgSize,count, False)
print ("train data:" , x_train.shape,  y_train.shape , clinical_train.shape ) 

print ("zeros: " , zeros , "ones: " , ones)
zeroWeight = ones / ((ones+zeros)*1.0)
oneWeight = zeros / ((ones+zeros)*1.0)
print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)



with tf.device('/gpu:0'):

    histories = funcs.Histories()

    model_0 = funcs.makeClinicalModel()

    model = Sequential()



    if fork:

        if mode == "3d":
            model_A = funcs.make3dConvModel(imgSize,count)  
            model_S = funcs.make3dConvModel(imgSize,count)  
            model_C = funcs.make3dConvModel(imgSize,count) 
        elif mode == "2d":
            model_A = funcs.make2dConvModel(imgSize)  
            model_S = funcs.make2dConvModel(imgSize)  
            model_C = funcs.make2dConvModel(imgSize)     

        # 
        model.add(keras.engine.topology.Merge([  model_0 , model_A, model_S, model_C  ], mode='concat', concat_axis=1)) # 512*3 + 3

    else:

        singleModel = funcs.makeSingle3dConvModel(imgSize, skip)

        model.add(keras.engine.topology.Merge([  model_0 , singleModel ], mode='concat', concat_axis=1)) # 512*3 + 3


    model.add(Dense(512, init = "he_normal" ))
    model.add(Dense(nb_classes, init = "he_normal" ))
    model.add(Activation('softmax'))

    # myOptimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    myOptimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # myOptimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0) 
    # myOptimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #
    model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])

    # save model
    plot(model, show_shapes=True , show_layer_names=True,  to_file='/home/ubuntu/output/' + RUN + '_model.png')


    # center and standardize - at this point its just the cubes
    mean,std,x_train_cs = funcs.centerAndStandardizeTraining(x_train)

    # pass back to funcs
    funcs.mean = mean
    funcs.std = std

    print ( "params: " , model.count_params() )


    if fork:

        model.fit_generator( krs.myGenerator(x_train_cs,y_train,clinical_train,finalSize,imgSize,count,batch_size) ,
                    samples_per_epoch= ( x_train_cs.shape[0] - (x_train_cs.shape[0]%batch_size) ) ,
                    class_weight={0 : zeroWeight, 1: oneWeight},
                    nb_epoch=nb_epoch,
                   callbacks=[histories])

    else:

        model.fit_generator( krs.myGenerator_single3D(x_train_cs,y_train,clinical_train,finalSize,imgSize,batch_size, skip) ,
                    samples_per_epoch= ( x_train_cs.shape[0] - (x_train_cs.shape[0]%batch_size) ) ,
                    # class_weight={0 : zeroWeight, 1: oneWeight},
                    nb_epoch=nb_epoch,
                   callbacks=[histories])










