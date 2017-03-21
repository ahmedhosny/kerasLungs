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
RUN = "25"
print (" training : run: A " , RUN)
mode = "2d"
batch_size = 64 # # 64 # 128
nb_classes = 2
nb_epoch = 2000
lr = 0.0001 #  no: 0.01, 0.001
count = 3 # for 3d mode, no i=og images to take in every direction
finalSize = 150 # from 150 down to.. 150
imgSize = 120 # 120
valTestMultiplier = 1
krs.augmentTraining = True

# for single3d
fork = True
# if fork false:
skip = 2 # (imgSize/skip should be int)
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


dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames("2yr")



#
x_train,y_train,zeros,ones,clinical_train =  funcs.getXandY(dataFrameTrain,imgSize, False)
print ("train data:" , x_train.shape,  y_train.shape  ) 
#
print ("zeros: " , zeros , "ones: " , ones)
zeroWeight = ones / ((ones+zeros)*1.0)
oneWeight = zeros / ((ones+zeros)*1.0)
print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)
funcs.zeroWeight = zeroWeight
funcs.oneWeight = oneWeight


with tf.device('/gpu:0'):

    histories = funcs.Histories()

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
        model.add(keras.engine.topology.Merge([ model_A, model_S, model_C  ], mode='concat', concat_axis=1)) # 512*3 + 3 # model_0 ,
        model.add(Dense(512))

    else:
         # overwrites model
        model = funcs.makeSingle3dConvModel(imgSize, skip)
        model.add(Dense(256))

    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    myOptimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])

    # save model
    plot(model, show_shapes=True , show_layer_names=True,  to_file='/home/ubuntu/output/' + RUN + '_model.png')

    # center and standardize - at this point its just the cubes
    mean,std,x_train_cs = funcs.centerAndStandardizeTraining(x_train)
    funcs.mean = mean
    funcs.std = std
    print ( "mean and std shape: " ,mean.shape,std.shape )

    print ( "params: " , model.count_params() )


    if fork:
        model.fit_generator( krs.myGenerator(x_train_cs,y_train,finalSize,imgSize,count,batch_size,mode) ,
                    samples_per_epoch= ( x_train_cs.shape[0] - (x_train_cs.shape[0]%batch_size) ) ,
                    class_weight={0 : zeroWeight, 1: oneWeight},
                    nb_epoch=nb_epoch,
                   callbacks=[histories])


    else:
        model.fit_generator( krs.myGenerator_single3D(x_train_cs,y_train,finalSize,imgSize,batch_size, skip) , 
                    samples_per_epoch= ( x_train_cs.shape[0] - (x_train_cs.shape[0]%batch_size) ) ,
                    class_weight={0 : zeroWeight, 1: oneWeight},
                    nb_epoch=nb_epoch,
                   callbacks=[histories])










