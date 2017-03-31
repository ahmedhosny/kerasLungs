from __future__ import division
import funcs
import krs
import numpy as np
import keras
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras import regularizers


# 2d + fork = axial,saggittal,coronal ( imgSize x imgSize )
# 3d + fork = axial,saggittal,coronal ( count*2+1 x imgSize x imgSize )
# 2d + no fork = axial ( imgSize x imgSize )
# 3d + no fork = cube ( imgSize/skip,imgSize/skip,imgSize/skip )


# current version
RUN = "48" 

# you want 2d or 3d convolutions?
mode = "2d"

# you want single architecture or 3-way architecture
fork = False

# final size should not be greater than 150
finalSize = 120 

# size of minipatch fed to net
imgSize = 80

# for 3d + fork , # of slices to take in each direction
count = 3

# for 3d + fork : number of slices to skip in that direction (2 will take every other slice) - can be any number
# for 3d + no fork : number of slices to skip across the entire cube ( should be imgSize%skip == 0  )
skip = 2

# augment while training?
# random minipatch is done regardless. This bool controls flipping and rotation
krs.augmentTraining = True

regul = regularizers.l2(0.0000001)

# others...
batch_size = 32 
nb_epoch = 300000
lr = 0.00001 

# print 
print ("training : run: " , RUN , " lr: " , lr)


#
#
#     `7MMF' .g8"""bgd `7MN.   `7MF' .g8""8q. `7MM"""Mq.  `7MM"""YMM
#       MM .dP'     `M   MMN.    M .dP'    `YM. MM   `MM.   MM    `7
#       MM dM'       `   M YMb   M dM'      `MM MM   ,M9    MM   d
#       MM MM            M  `MN. M MM        MM MMmmdM9     MMmmMM
#       MM MM.    `7MMF' M   `MM.M MM.      ,MP MM  YM.     MM   Y  ,
#       MM `Mb.     MM   M     YMM `Mb.    ,dP' MM   `Mb.   MM     ,M
#     .JMML. `"bmmmdPY .JML.    YM   `"bmmd"' .JMML. .JMM..JMMmmmmMMM
#

nb_classes = 2
funcs.RUN = RUN
funcs.mode = mode
funcs.imgSize = imgSize
funcs.count = count
funcs.finalSize = finalSize
funcs.fork = fork
funcs.skip = skip



#
#
#     `7MM"""Yb.      db   MMP""MM""YMM   db
#       MM    `Yb.   ;MM:  P'   MM   `7  ;MM:
#       MM     `Mb  ,V^MM.      MM      ,V^MM.
#       MM      MM ,M  `MM      MM     ,M  `MM
#       MM     ,MP AbmmmqMA     MM     AbmmmqMA
#       MM    ,dP'A'     VML    MM    A'     VML
#     .JMMmmmdP'.AMA.   .AMMA..JMML..AMA.   .AMMA.
#
#

#1# get dataframnes
dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames()

#2# get data
x_train,y_train,zeros,ones =  funcs.getXandY(dataFrameTrain,imgSize)
print ("train data:" , x_train.shape,  y_train.shape  ) 
#
print ("zeros: " , zeros , "ones: " , ones)
zeroWeight = ones / ((ones+zeros)*1.0)
oneWeight = zeros / ((ones+zeros)*1.0)
print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)
funcs.zeroWeight = zeroWeight
funcs.oneWeight = oneWeight


with tf.device('/gpu:0'):


    #
    #
    #     `7MMM.     ,MMF' .g8""8q. `7MM"""Yb. `7MM"""YMM  `7MMF'
    #       MMMb    dPMM .dP'    `YM. MM    `Yb. MM    `7    MM
    #       M YM   ,M MM dM'      `MM MM     `Mb MM   d      MM
    #       M  Mb  M' MM MM        MM MM      MM MMmmMM      MM
    #       M  YM.P'  MM MM.      ,MP MM     ,MP MM   Y  ,   MM      ,
    #       M  `YM'   MM `Mb.    ,dP' MM    ,dP' MM     ,M   MM     ,M
    #     .JML. `'  .JMML. `"bmmd"' .JMMmmmdP' .JMMmmmmMMM .JMMmmmmMMM
    #
    #


    histories = funcs.Histories()

    
    if fork:

        model = Sequential()

        if mode == "3d":
            model_A = funcs.make3dConvModel(imgSize,count,fork,skip)  
            model_S = funcs.make3dConvModel(imgSize,count,fork,skip)  
            model_C = funcs.make3dConvModel(imgSize,count,fork,skip) 

        elif mode == "2d":
            model_A = funcs.make2dConvModel(imgSize,regul)  
            model_S = funcs.make2dConvModel(imgSize,regul)  
            model_C = funcs.make2dConvModel(imgSize,regul)     

        # 
        model.add(keras.engine.topology.Merge([ model_A, model_S, model_C  ], mode='concat', concat_axis=1)) #  output here is 512*3 
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    else:

        if mode == "3d":
            model = funcs.make3dConvModel(imgSize, count ,fork,skip) # output here is 512

        elif mode == "2d":
            model = funcs.make2dConvModel(imgSize,regul) # output here is 512

        model.add(Dense(256 , activity_regularizer = regul ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))


    # add last dense and softmax
    model.add(Dense(nb_classes , activity_regularizer = regul ))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    myOptimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=myOptimizer, metrics=['accuracy'])

    # save model
    plot(model, show_shapes=True , show_layer_names=True,  to_file='/home/ubuntu/output/' + RUN + '_model.png')



    #
    #
    #     `7MN.   `7MF' .g8""8q. `7MM"""Mq.  `7MMM.     ,MMF'      db      `7MMF'      `7MMF'MMM"""AMV `7MM"""YMM
    #       MMN.    M .dP'    `YM. MM   `MM.   MMMb    dPMM       ;MM:       MM          MM  M'   AMV    MM    `7
    #       M YMb   M dM'      `MM MM   ,M9    M YM   ,M MM      ,V^MM.      MM          MM  '   AMV     MM   d
    #       M  `MN. M MM        MM MMmmdM9     M  Mb  M' MM     ,M  `MM      MM          MM     AMV      MMmmMM
    #       M   `MM.M MM.      ,MP MM  YM.     M  YM.P'  MM     AbmmmqMA     MM      ,   MM    AMV   ,   MM   Y  ,
    #       M     YMM `Mb.    ,dP' MM   `Mb.   M  `YM'   MM    A'     VML    MM     ,M   MM   AMV   ,M   MM     ,M
    #     .JML.    YM   `"bmmd"' .JMML. .JMM..JML. `'  .JMML..AMA.   .AMMA..JMMmmmmMMM .JMML.AMVmmmmMM .JMMmmmmMMM
    #
    #

    # center and standardize - at this point its just the cubes
    mean,std,x_train_cs = funcs.centerAndStandardizeTraining(x_train)
    funcs.mean = mean
    funcs.std = std

    print ( "mean and std shape: " ,mean.shape,std.shape )

    np.save( "/home/ubuntu/output/" + RUN + "_mean.npy", mean)
    np.save( "/home/ubuntu/output/" + RUN + "_std.npy", std)

    print ( "params: " , model.count_params() )



    #
    #
    #     `7MM"""YMM `7MMF'MMP""MM""YMM
    #       MM    `7   MM  P'   MM   `7
    #       MM   d     MM       MM
    #       MM""MM     MM       MM
    #       MM   Y     MM       MM
    #       MM         MM       MM
    #     .JMML.     .JMML.   .JMML.
    #
    #

    # fit the model
    model.fit_generator( krs.myGenerator(x_train_cs,y_train,finalSize,imgSize,count,batch_size,mode,fork,skip) ,
                    samples_per_epoch= ( x_train_cs.shape[0] - (x_train_cs.shape[0]%batch_size) ) ,
                    # class_weight={0 : zeroWeight, 1: oneWeight},
                    nb_epoch=nb_epoch,
                   callbacks=[histories])












