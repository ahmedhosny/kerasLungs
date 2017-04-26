
from __future__ import division
from __future__ import print_function

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

import krs
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , advanced_activations
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D , MaxPooling3D
from sklearn.metrics import roc_auc_score
import time
from keras import backend as K
import random
from tensorflow.python.ops import nn
from keras.layers.normalization import BatchNormalization
K.set_image_dim_ordering('tf')


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


# run 50   
def manageDataFramesEqually():
    trainList = ["nsclc_rt"] 
    validateList = ["lung1"] 
    testList = ["lung2"]

    dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
    dataFrame = dataFrame [ 
        ( pd.notnull( dataFrame["pathToData"] ) ) &
        ( pd.notnull( dataFrame["pathToMask"] ) ) &
        ( pd.notnull( dataFrame["stackMin"] ) ) &
        ( pd.isnull( dataFrame["patch_failed"] ) ) &
        # ( pd.notnull( dataFrame["surv1yr"] ) )  &
        ( pd.notnull( dataFrame["surv2yr"] ) )  &
        ( pd.notnull( dataFrame["histology_grouped"] ) ) # &
        # ( pd.notnull( dataFrame["stage"] ) ) 
        # ( pd.notnull( dataFrame["age"] ) )  
        ]
   
    dataFrame = dataFrame.reset_index(drop=True)
    
    ###### FIX ALL
    
    #1# clean histology - remove smallcell and other
    # histToInclude - only NSCLC
    histToInclude = [1.0,2.0,3.0,4.0]
    # not included - SCLC and other and no data [ 0,5,6,7,8,9 ]
    dataFrame = dataFrame [ dataFrame.histology_grouped.isin(histToInclude) ]
    dataFrame = dataFrame.reset_index(drop=True)

    
    #2# use 1,2,3 stages
    # stageToInclude = [1.0,2.0,3.0]
    # dataFrame = dataFrame [ dataFrame.stage.isin(stageToInclude) ]
    # dataFrame = dataFrame.reset_index(drop=True)
    # print ("all patients: " , dataFrame.shape)

        
    ###### GET TRAINING  

    dataFrameTrain = dataFrame [ dataFrame["dataset"].isin(trainList) ]
    #3# type of treatment - use only radio or chemoRadio - use .npy file
    chemoRadio = np.load("rt_chemoRadio.npy").astype(str)
    dataFrameTrain = dataFrameTrain [ dataFrameTrain["patient"].isin(chemoRadio) ]
    #4# (rt only) use all causes of death
    # not implemented
    dataFrameTrain = dataFrameTrain.reset_index(drop=True)
#     print ("train patients " , dataFrameTrain.shape)
    
    #### GET VAL
    dataFrameValidate = dataFrame [ dataFrame["dataset"].isin(validateList) ]
    dataFrameValidate = dataFrameValidate.reset_index(drop=True)
#     print ("validate patients : " , dataFrameValidate.shape)   
    
    ##### GET TEST
    dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
    dataFrameTest = dataFrameTest.reset_index(drop=True)
#     print ("test size : " , dataFrameTest.shape)
    
    # put all, shuffle then reset index
    dataFrame = pd.concat ( [  dataFrameTrain , dataFrameValidate , dataFrameTest  ]   )
    dataFrame = dataFrame.sample( frac=1 , random_state = 245 ) # this random seed gives ok class balance in training
    dataFrame = dataFrame.reset_index(drop=True)
#     print ("all together : " , dataFrame.shape) 
    
    # split
    
    dataFrameTrain, dataFrameValidate, dataFrameTest = np.split(dataFrame,
                                                                [int(.75*len(dataFrame)), int(.83*len(dataFrame))])
    
    
    dataFrameTrain = dataFrameTrain.reset_index(drop=True)
    print (  "zeros: " , len( [ x for x in dataFrameTrain.surv2yr.tolist() if x == 0.0  ] )   ) 
    print (  "ones: " , len( [ x for x in dataFrameTrain.surv2yr.tolist() if x == 1.0  ] )   )   
    print ("train patients " , dataFrameTrain.shape)
    
    #### GET VAL
    dataFrameValidate = dataFrameValidate.reset_index(drop=True)
    print (  "zeros: " , len( [ x for x in dataFrameValidate.surv2yr.tolist() if x == 0.0  ] )   ) 
    print (  "ones: " , len( [ x for x in dataFrameValidate.surv2yr.tolist() if x == 1.0  ] )   ) 
    print ("validate patients : " , dataFrameValidate.shape)   
    
    ##### GET TEST
    dataFrameTest = dataFrameTest.reset_index(drop=True)
    print (  "zeros: " , len( [ x for x in dataFrameTest.surv2yr.tolist() if x == 0.0  ] )   ) 
    print (  "ones: " , len( [ x for x in dataFrameTest.surv2yr.tolist() if x == 1.0  ] )   ) 
    print ("test size : " , dataFrameTest.shape)
    
    

    return dataFrameTrain, dataFrameValidate,dataFrameTest




def manageDataFrames():
    trainList = ["nsclc_rt"]  # , , , ,  ,"oncopanel" , "moffitt","moffittSpore"  ,"oncomap" , ,"lung3" 
    validateList = ["lung2"] 
    testList = ["lung1"] # split to val and test

    dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
    dataFrame = dataFrame [ 
        ( pd.notnull( dataFrame["pathToData"] ) ) &
        ( pd.notnull( dataFrame["pathToMask"] ) ) &
        ( pd.notnull( dataFrame["stackMin"] ) ) &
        ( pd.isnull( dataFrame["patch_failed"] ) ) &
        # ( pd.notnull( dataFrame["surv1yr"] ) )  &
        ( pd.notnull( dataFrame["surv2yr"] ) )  &
        ( pd.notnull( dataFrame["histology_grouped"] ) )  &
        ( pd.notnull( dataFrame["stage"] ) ) 
        # ( pd.notnull( dataFrame["age"] ) )  
        ]
   
    dataFrame = dataFrame.reset_index(drop=True)
    
    ###### FIX ALL
    
    #1# clean histology - remove smallcell and other
    # histToInclude - only NSCLC
    histToInclude = [1.0,2.0,3.0,4.0]
    # not included - SCLC and other and no data [ 0,5,6,7,8,9 ]
    dataFrame = dataFrame [ dataFrame.histology_grouped.isin(histToInclude) ]
    dataFrame = dataFrame.reset_index(drop=True)

    
    # #2# use 1,2,3 stages no 1
    stageToInclude = [1.0,2.0,3.0]
    dataFrame = dataFrame [ dataFrame.stage.isin(stageToInclude) ]
    dataFrame = dataFrame.reset_index(drop=True)
    print ("all patients: " , dataFrame.shape)

        
    ###### GET TRAINING / VALIDATION 

    dataFrameTrain = dataFrame [ dataFrame["dataset"].isin(trainList) ]
    #3# type of treatment - use only radio or chemoRadio - use .npy file
    chemoRadio = np.load("rt_chemoRadio.npy").astype(str)
    dataFrameTrain = dataFrameTrain [ dataFrameTrain["patient"].isin(chemoRadio) ]
    #4# (rt only) use all causes of death
    # not implemented
    dataFrameTrain = dataFrameTrain.reset_index(drop=True)
    print ("train patients " , dataFrameTrain.shape)

    # Val
    dataFrameValidate = dataFrame [ dataFrame["dataset"].isin(validateList) ]
    dataFrameValidate = dataFrameValidate.reset_index(drop=True)
    print ("final - val size : " , dataFrameValidate.shape)

    # TEST
    dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
    dataFrameTest = dataFrameTest.reset_index(drop=True)
    print ("final - test size : " , dataFrameTest.shape)


    return dataFrameTrain,dataFrameValidate,dataFrameTest 

# def manageDataFrames():
#     trainList = ["nsclc_rt"]  # , , , ,  ,"oncopanel" , "moffitt","moffittSpore"  ,"oncomap" , ,"lung3" 
#     validateList = ["lung2"] # leave empty
#     testList = ["lung1"] # split to val and test

#     dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
#     dataFrame = dataFrame [ 
#         ( pd.notnull( dataFrame["pathToData"] ) ) &
#         ( pd.notnull( dataFrame["pathToMask"] ) ) &
#         ( pd.notnull( dataFrame["stackMin"] ) ) &
#         ( pd.isnull( dataFrame["patch_failed"] ) ) &
#         # ( pd.notnull( dataFrame["surv1yr"] ) )  &
#         ( pd.notnull( dataFrame["surv2yr"] ) )  &
#         ( pd.notnull( dataFrame["histology_grouped"] ) ) #  &
#         ( pd.notnull( dataFrame["stage"] ) ) 
#         # ( pd.notnull( dataFrame["age"] ) )  
#         ]
   
#     dataFrame = dataFrame.reset_index(drop=True)
    
#     ###### FIX ALL
    
#     #1# clean histology - remove smallcell and other
#     # histToInclude - only NSCLC
#     histToInclude = [1.0,2.0,3.0,4.0]
#     # not included - SCLC and other and no data [ 0,5,6,7,8,9 ]
#     dataFrame = dataFrame [ dataFrame.histology_grouped.isin(histToInclude) ]
#     dataFrame = dataFrame.reset_index(drop=True)

    
#     # #2# use 1,2,3 stages no 1
#     stageToInclude = [1.0,2.0,3.0]
#     dataFrame = dataFrame [ dataFrame.stage.isin(stageToInclude) ]
#     dataFrame = dataFrame.reset_index(drop=True)
#     print ("all patients: " , dataFrame.shape)

        
#     ###### GET TRAINING / VALIDATION 

#     dataFrameTrain = dataFrame [ dataFrame["dataset"].isin(trainList) ]
#     #3# type of treatment - use only radio or chemoRadio - use .npy file
#     chemoRadio = np.load("rt_chemoRadio.npy").astype(str)
#     dataFrameTrain = dataFrameTrain [ dataFrameTrain["patient"].isin(chemoRadio) ]
#     #4# (rt only) use all causes of death
#     # not implemented
#     dataFrameTrain = dataFrameTrain.reset_index(drop=True)
#     print ("train patients " , dataFrameTrain.shape)

#     dataFrameValidate = dataFrame [ dataFrame["dataset"].isin(validateList) ]
#     dataFrameValidate = dataFrameValidate.reset_index(drop=True)
#     print ("validate patients : " , dataFrameValidate.shape)


#     #
#     # now combine train and val , then split them.
#     dataFrameTrainValidate = pd.concat([dataFrameTrain,dataFrameValidate] , ignore_index=False )
#     dataFrameTrainValidate = dataFrameTrainValidate.sample( frac=1 , random_state = 42 )
#     dataFrameTrainValidate = dataFrameTrainValidate.reset_index(drop=True)
#     print ("final - train and validate patients : " , dataFrameTrainValidate.shape)


#     thirty = int(dataFrameTrainValidate.shape[0]*0.06)   ######################################
#     if thirty % 2 != 0:
#         thirty = thirty + 1



#     # get 0's and 1's.
#     zero = dataFrameTrainValidate [  (dataFrameTrainValidate['surv2yr']== 0.0)  ]
#     one = dataFrameTrainValidate [  (dataFrameTrainValidate['surv2yr']== 1.0)  ]

#     print ( zero.shape , one.shape )
#     # split to train and val
#     half = int(thirty/2.0)

#     trueList = [True for i in range (half)]

#     #
#     zeroFalseList = [False for i in range (zero.shape[0] - half )]
#     zero_msk = trueList + zeroFalseList
#     random.seed(41)
#     random.shuffle(zero_msk)
#     zero_msk = np.array(zero_msk)
#     #
#     oneFalseList = [False for i in range (one.shape[0] - half )]
#     one_msk = trueList + oneFalseList
#     random.seed(41)
#     random.shuffle(one_msk)
#     one_msk = np.array(one_msk)



#     # TRAIN
#     zero_train = zero[~zero_msk]
#     one_train = one[~one_msk]
#     dataFrameTrain = pd.DataFrame()
#     dataFrameTrain = dataFrameTrain.append( zero_train )  #.sample( frac=0.73 , random_state = 42 ) 
#     dataFrameTrain = dataFrameTrain.append(one_train)
#     dataFrameTrain = dataFrameTrain.sample( frac=1 , random_state = 42 )
#     dataFrameTrain = dataFrameTrain.reset_index(drop=True)
#     print ('final - train size:' , dataFrameTrain.shape)


#     # VALIDATE
#     zero_val = zero[zero_msk]
#     one_val = one[one_msk]
#     dataFrameValidate = pd.DataFrame()
#     dataFrameValidate = dataFrameValidate.append(zero_val)
#     dataFrameValidate = dataFrameValidate.append(one_val)
#     dataFrameValidate = dataFrameValidate.sample( frac=1 , random_state = 42 )
#     dataFrameValidate = dataFrameValidate.reset_index(drop=True)
#     print ('final - validate size:' , dataFrameValidate.shape)


#     # TEST
#     dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
#     dataFrameTest = dataFrameTest.reset_index(drop=True)
#     print ("final - test size : " , dataFrameTest.shape)

    

#     return dataFrameTrain,dataFrameValidate,dataFrameTest 


# used for evaluating performance 
def aggregate(logits,mul):
    logitsOut = []
    #
    for i in range ( 0,logits.shape[0],mul ):
        tempVal0 = 0
        tempVal1 = 0
        for k in range (mul):
            tempVal0 += logits[i+k][0]
            tempVal1 += logits[i+k][1]
        val0 = tempVal0 / (mul*1.0)
        val1 = tempVal1 / (mul*1.0)
        logitsOut.append( [ val0,val1 ] )
    #
    return np.array(logitsOut)


def getXandY(dataFrame,imgSize):



    arrList = []
    y = []
    zeros = 0
    ones = 0
    # clincical = []
    
    for i in range (dataFrame.shape[0]):

        npy =  "/home/ahmed/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"
        # npy =  "~/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"
        arr = np.load(npy)

        # X #
        arrList.append (  arr )  

        # Y #
        if whatToPredict == "survival":
            y.append ( int(dataFrame.surv2yr[i])  )
        elif whatToPredict == "stage":
            y.append ( int(dataFrame.stage[i])  )
        elif whatToPredict == "histology":
            y.append ( int(dataFrame.histology_grouped[i])  )


        # zeros and ones
        if int(dataFrame.surv2yr[i]) == 1:
            ones = ones+1
        elif int(dataFrame.surv2yr[i]) == 0:
            zeros = zeros+1
        else:
            raise Exception("a survival value is not 0 or 1")

        # # now clinical
        # clincicalVector = [ dataFrame.age[i] , dataFrame.stage[i] , dataFrame.histology_grouped[i] ]
        # clincical.extend( [clincicalVector for x in range(1)] )


    # after loop
    arrList = np.array(arrList, 'float32')
    y = np.array(y, 'int8')
    y = np_utils.to_categorical(y, NUMCLASSES)  
    # clincical = np.array(clincical , 'float32'  )
    return arrList,y,zeros,ones # ,clincical

def getX(dataFrame,imgSize):

    arrList = []

    for i in range (dataFrame.shape[0]):

        npy =  "/home/ahmed/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"

        arr = np.load(npy)

        # X #
        arrList.append (  arr )  

    # after loop
    arrList = np.array(arrList, 'float32')

    return arrList



#
#
#     `7MMF' `YMM' `7MM"""YMM  `7MM"""Mq.        db       .M"""bgd     `7MMM.     ,MMF' .g8""8q. `7MM"""Yb. `7MM"""YMM  `7MMF'
#       MM   .M'     MM    `7    MM   `MM.      ;MM:     ,MI    "Y       MMMb    dPMM .dP'    `YM. MM    `Yb. MM    `7    MM
#       MM .d"       MM   d      MM   ,M9      ,V^MM.    `MMb.           M YM   ,M MM dM'      `MM MM     `Mb MM   d      MM
#       MMMMM.       MMmmMM      MMmmdM9      ,M  `MM      `YMMNq.       M  Mb  M' MM MM        MM MM      MM MMmmMM      MM
#       MM  VMA      MM   Y  ,   MM  YM.      AbmmmqMA   .     `MM       M  YM.P'  MM MM.      ,MP MM     ,MP MM   Y  ,   MM      ,
#       MM   `MM.    MM     ,M   MM   `Mb.   A'     VML  Mb     dM       M  `YM'   MM `Mb.    ,dP' MM    ,dP' MM     ,M   MM     ,M
#     .JMML.   MMb..JMMmmmmMMM .JMML. .JMM..AMA.   .AMMA.P"Ybmmd"      .JML. `'  .JMML. `"bmmd"' .JMMmmmdP' .JMMmmmmMMM .JMMmmmmMMM
#
#

# not used
def makeClinicalModel():
    model = Sequential()
    # just histology, stage and age
    model.add(Dense( 3, input_dim=(3)) ) # 512
    return model

def make2dConvModel(imgSize,regul):
    # regul - norm - act
    #(samples, rows, cols, channels) if dim_ordering='tf'.

    model = Sequential()

    model.add(Convolution2D(48, 7, 7,  border_mode='valid', dim_ordering='tf', input_shape=[imgSize,imgSize,1] , activity_regularizer = regul )) # 32
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    # model.add(MaxPooling2D(pool_size=(3, 3)  )) 


    model.add(Convolution2D(96, 5, 5 ,  border_mode='valid', activity_regularizer = regul )) # 32
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    model.add(MaxPooling2D(pool_size=(3, 3) ))


#     model.add(Convolution2D(192, 3, 3 ,  border_mode='valid' , activity_regularizer = regul )) # 64
#     model.add(BatchNormalization())
#     model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    model.add(Convolution2D(192, 3, 3 ,  border_mode='valid' , activity_regularizer = regul )) # 64
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))


    model.add(Convolution2D(256, 3, 3 ,  border_mode='valid' , activity_regularizer = regul )) # 64
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2) ))



    # model.add(BatchNormalization())
    # model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Dropout(0.25))

    # # # this chucnk added - 14
    # model.add(Convolution2D(256, 3, 3, border_mode='valid' , activity_regularizer = regul )) # 64
    # model.add(BatchNormalization())
    # model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    # model.add(Convolution2D(256, 3, 3,  border_mode='valid' , activity_regularizer = regul )) # 64
    # model.add(BatchNormalization())
    # model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512 , activity_regularizer = regul )) # 512
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))
    model.add(Dropout(0.5))

    
    return model


def make3dConvModel(imgSize,count,fork,skip,regul):
    #(samples, rows, cols, channels) if dim_ordering='tf'.
    
    convDrop = 0.25

    model = Sequential()

    if fork:
        model.add(Convolution3D(64, 1, 3, 3, border_mode='valid',dim_ordering='tf',input_shape=[count*2+1,imgSize,imgSize,1] , activity_regularizer = regul)) # 32
    else:
        model.add(Convolution3D(64, 1, 3, 3, border_mode='valid',dim_ordering='tf',input_shape=[imgSize/skip,imgSize/skip,imgSize/skip,1] , activity_regularizer = regul )) # 32
        # model.add(Convolution3D(48, 5, 5, 5, border_mode='valid',dim_ordering='tf',input_shape=[count*2+1,imgSize,imgSize,1] , activity_regularizer = regul)) # 32

    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))
    model.add(Dropout(convDrop))

    model.add(Convolution3D(96, 1, 3, 3 ,  border_mode='valid' , activity_regularizer = regul )) # 32
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    model.add(MaxPooling3D(pool_size=(1, 3, 3 ))) ### 
    model.add(Dropout(convDrop))
    
    model.add(Convolution3D(192, 1, 3, 3 ,  border_mode='valid' , activity_regularizer = regul )) # 32
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))
    model.add(Dropout(convDrop))
    
    model.add(Convolution3D(384, 1, 3, 3 ,  border_mode='valid' , activity_regularizer = regul )) # 32
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))

    model.add(MaxPooling3D(pool_size=(1, 3, 3 ))) ### 
    model.add(Dropout(convDrop))


    model.add(Flatten())
    model.add(Dense(512 , activity_regularizer = regul )) # 512
    model.add(BatchNormalization())
    model.add(advanced_activations.LeakyReLU(alpha=LRELUalpha))
    model.add(Dropout(0.5))
    
    return model





#
#
#     `7MMF' `YMM' `7MM"""YMM  `7MM"""Mq.        db       .M"""bgd     `7MMF'  `7MMF'`7MMF' .M"""bgd MMP""MM""YMM
#       MM   .M'     MM    `7    MM   `MM.      ;MM:     ,MI    "Y       MM      MM    MM  ,MI    "Y P'   MM   `7
#       MM .d"       MM   d      MM   ,M9      ,V^MM.    `MMb.           MM      MM    MM  `MMb.          MM
#       MMMMM.       MMmmMM      MMmmdM9      ,M  `MM      `YMMNq.       MMmmmmmmMM    MM    `YMMNq.      MM
#       MM  VMA      MM   Y  ,   MM  YM.      AbmmmqMA   .     `MM       MM      MM    MM  .     `MM      MM
#       MM   `MM.    MM     ,M   MM   `Mb.   A'     VML  Mb     dM       MM      MM    MM  Mb     dM      MM
#     .JMML.   MMb..JMMmmmmMMM .JMML. .JMM..AMA.   .AMMA.P"Ybmmd"      .JMML.  .JMML..JMML.P"Ybmmd"     .JMML.
#
#

# new normalization method
# get mean and std from training
def centerAndNormalize(arr):
    # out = arr
    # #

    # out -= 1000.0
    # out /= 2000.0
    #

    out = arr

    oldMin = -1024
    oldRange = 3071+1024

    newRange = 1
    newMin = 0

    sikoAll = ((( out  - oldMin) * newRange) / oldRange) + newMin

    return sikoAll

# get mean and std from training
def centerAndStandardizeTraining(arr):
    out = arr
    #
    mean = np.mean(out,axis=(0) )
    std = np.std(out,axis=(0) )
    # 
    out -= mean
    out /= (std + np.finfo(float).eps )
    #
    return mean,std,out

# apply mean and std to val and test
def centerAndStandardizeValTest(arr,mean,std):
    out = arr
    #
    out -= mean
    out /= (std + np.finfo(float).eps )
    #
    return out


def AUC(test_labels,test_prediction,nb):

    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 

def AUCalt( test_labels , test_prediction):
    # convert to non-categorial
    test_prediction = np.array( [ x[1] for x in test_prediction   ])
    test_labels = np.array( [ 0 if x[0]==1 else 1 for x in test_labels   ])
    # get rates
    fpr, tpr, thresholds = roc_curve(test_labels, test_prediction, pos_label=1)
    # get auc
    myAuc = auc(fpr, tpr)
    return myAuc



class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):

        self.train_loss = []
        self.auc = []
        self.logits = []
        self.val_loss = []

        # save json representation
        model_json = self.model.to_json()
        with open("/home/ahmed/output/" + RUN + "_json.json", "w") as json_file:
            json_file.write(model_json)


        dataFrameTrain,dataFrameValidate,dataFrameTest= manageDataFrames()
        #
        x_val,y_val,zeros,ones =  getXandY(dataFrameValidate,imgSize)
        print ("validation data:" , x_val.shape,  y_val.shape , zeros , ones ) 
        self.dataFrameValidate = dataFrameValidate
        self.y_val = y_val
        # lets do featurewiseCenterAndStd - its still a cube at this point
        # x_val_cs = centerAndStandardizeValTest(x_val,mean,std)
        x_val_cs = centerAndNormalize(x_val)
        # x_val_cs = x_val


        if fork:
            # lets get the 3 orientations
            self.x_val_a,self.x_val_s,self.x_val_c = krs.splitValTest(x_val_cs,finalSize,imgSize,count,mode,fork,skip)
            print ("final val data:" , self.x_val_a.shape,self.x_val_s.shape,self.x_val_c.shape)

        else:
            # lets get one only
            self.x_val = krs.splitValTest(x_val_cs,finalSize,imgSize,count,mode,fork,skip)
            print ("final val data:" , x_val.shape)

        return


    def on_train_end(self, logs={}):

        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        # # val_loss__ =  self.model.test_on_batch ( [ self.x_val ] , self.y_val )[0]
        # val_loss_ =  self.model.evaluate ( [ self.x_val ] , self.y_val , batch_size = self.dataFrameValidate.shape[0]  )[0]

        

        # if epoch > 300:
        #     if all(val_loss_< i for i in self.val_loss):
        #         self.model.save_weights("/home/ahmed/output/" + RUN + "_model.h5")
        #         print("Saved model to disk")
        #         # save model and json representation
        #         model_json = self.model.to_json()
        #         with open("/home/ahmed/output/" + RUN + "_json.json", "w") as json_file:
        #             json_file.write(model_json)

        # # append and save train loss
        # self.train_loss.append(logs.get('loss'))
        # np.save( "/home/ahmed/output/" + RUN + "_train_loss.npy", self.train_loss)

        # # append and save train loss
        # self.val_loss.append(val_loss_)
        # np.save( "/home/ahmed/output/" + RUN + "_val_loss.npy", self.val_loss) 

        

        # print ( "val loss: " , val_loss_  )
        # print ( "val loss: " , val_loss_  )


        # logits = self.model.predict ( [ self.x_val ] )

        # print ( "\npredicted val zeros: "  , len( [ x for x in  logits if x[0] > x[1]  ] )  )
        # print ( "predicted val ones: "  , len( [ x for x in  logits if x[0] < x[1]  ] )  )

        # logits = np.array(logits)


        # print ("logits: " , logits.shape , logits[0]    )
        # auc1 , auc2 = AUC(  self.y_val ,  logits )
        # print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
        # print ("wtf2")


        # # append and save auc
        # self.auc.append(auc1)
        # np.save( "/home/ahmed/output/" + RUN + "_auc.npy", self.auc)

        # # append and save logits
        # self.logits.append(logits)
        # np.save( "/home/ahmed/output/" + RUN + "_logits.npy", self.logits)

        ###############################################################################################################

        logits = []

        for i in range (self.dataFrameValidate.shape[0]):

            if fork: 

                if mode == "3d":
                    # get predictions
                    y_pred = self.model.predict_on_batch ( [ self.x_val_a[i].reshape(1,count*2+1,imgSize,imgSize,1) , 
                        self.x_val_s[i].reshape(1,count*2+1,imgSize,imgSize,1) , 
                        self.x_val_c[i].reshape(1,count*2+1,imgSize,imgSize,1) ]  )

                elif mode == "2d":
                    # get predictions
                    y_pred = self.model.predict_on_batch ( [ self.x_val_a[i].reshape(1,imgSize,imgSize,1) ,
                        self.x_val_s[i].reshape(1,imgSize,imgSize,1) , 
                        self.x_val_c[i].reshape(1,imgSize,imgSize,1) ]  )

            else:

                if mode == "3d":
                    # get predictions
                    dim = int ( imgSize/( 1.0* skip) )
                    y_pred = self.model.predict_on_batch ( [ self.x_val[i].reshape(1,dim,dim,dim,1) ] ) 
                    # y_pred = self.model.predict_on_batch ( [ self.x_val[i].reshape(1,count*2+1,imgSize,imgSize,1) ] ) 

                elif mode == "2d":
                    # get predictions

                    y_pred = self.model.predict_on_batch ( [ self.x_val[i].reshape(1,imgSize,imgSize,1) ] )

            # now after down with switching
            logits.append( y_pred[0] )



        print ( "\npredicted val zeros: "  , len( [ x for x in  logits if x[0] > x[1]  ] )  )
        print ( "predicted val ones: "  , len( [ x for x in  logits if x[0] < x[1]  ] )  )

        logits = np.array(logits)

        print ("logits: " , logits.shape , logits[0]    )
        auc1 , auc2 = AUC(  self.y_val ,  logits , NUMCLASSES )

        print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
        print ("wtf2")

        # # before appending, check if this auc is the highest in all the list, if yes save the h5 model
        #
        if epoch > 10:
            if all(auc1>i for i in self.auc):
                self.model.save_weights("/home/ahmed/output/" + RUN + "_model.h5")
                print("Saved model to disk")
                # save model and json representation
                model_json = self.model.to_json()
                with open("/home/ahmed/output/" + RUN + "_json.json", "w") as json_file:
                    json_file.write(model_json)

        # append and save train loss
        self.train_loss.append(logs.get('loss'))
        np.save( "/home/ahmed/output/" + RUN + "_train_loss.npy", self.train_loss) 

        # append and save auc
        self.auc.append(auc1)
        np.save( "/home/ahmed/output/" + RUN + "_auc.npy", self.auc)

        # append and save logits
        self.logits.append(logits)
        np.save( "/home/ahmed/output/" + RUN + "_logits.npy", self.logits)
         
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):

        return


#
#
#     `7MMF' `YMM' `7MM"""YMM  `7MM"""YMM  `7MM"""Mq.
#       MM   .M'     MM    `7    MM    `7    MM   `MM.
#       MM .d"       MM   d      MM   d      MM   ,M9
#       MMMMM.       MMmmMM      MMmmMM      MMmmdM9
#       MM  VMA      MM   Y  ,   MM   Y  ,   MM
#       MM   `MM.    MM     ,M   MM     ,M   MM
#     .JMML.   MMb..JMMmmmmMMM .JMMmmmmMMM .JMML.
#
#


# define funcs

# if fork:

#     # (0 = test, 1 = train) 
#     axialFunc = K.function([ self.model.layers[0].layers[0].layers[0].input , K.learning_phase()  ], 
#                        [ self.model.layers[0].layers[0].layers[-1].output ] )

#     sagittalFunc = K.function([ self.model.layers[0].layers[1].layers[0].input , K.learning_phase()  ], 
#                        [ self.model.layers[0].layers[1].layers[-1].output ] )

#     coronalFunc = K.function([ self.model.layers[0].layers[2].layers[0].input , K.learning_phase()  ], 
#                        [ self.model.layers[0].layers[2].layers[-1].output ] )

#     mergeFunc = K.function([ self.model.layers[1].input , K.learning_phase()  ], 
#                        [ self.model.layers[2].output ] ) 

#     softmaxFunc = K.function([ self.model.layers[3].input , K.learning_phase()  ], 
#                        [ self.model.layers[3].output ] )

# else:

#     print("no fork - not tested")




# use funcs

# if mode == "2d":
#     # get the different ones
#     axial512 = axialFunc( [  self.x_val_a[i].reshape(1,imgSize,imgSize,1) , 0 ] )
#     sagittal512 = sagittalFunc( [  self.x_val_s[i].reshape(1,imgSize,imgSize,1) , 0 ] )
#     coronal512 = coronalFunc( [  self.x_val_c[i].reshape(1,imgSize,imgSize,1) , 0 ] )
# if mode == "3d":
#     axial512 = axialFunc( [  self.x_val_a[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ] )
#     sagittal512 = sagittalFunc( [  self.x_val_s[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ] )
#     coronal512 = coronalFunc( [  self.x_val_c[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ] )

# # concat them
# concat = []
# concat.extend ( axial512[0][0].tolist() )
# concat.extend ( sagittal512[0][0].tolist() )
# concat.extend ( coronal512[0][0].tolist() )
# #
# concat = np.array(concat ,'float32').reshape(1,len(concat))
# # now do one last function
# preds = mergeFunc( [ concat , 0 ])
# #
# logitsBal = np.array( [ preds[0][0][0]  ,  preds[0][0][1]  ]  ) .reshape(1,2)   # * zeroWeight -  * oneWeight
# logits.append(  softmaxFunc(   [ logitsBal     , 0 ]) [0].reshape(2)  )

# run 54
# def manageDataFramesLung1():
#     trainList = ["lung2"] 
#     testList = ["nsclc_rt"] # or lung1 # needs fixing

#     dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
#     dataFrame = dataFrame [ 
#         ( pd.notnull( dataFrame["pathToData"] ) ) &
#         ( pd.notnull( dataFrame["pathToMask"] ) ) &
#         ( pd.notnull( dataFrame["stackMin"] ) ) &
#         ( pd.isnull( dataFrame["patch_failed"] ) ) &
#         # ( pd.notnull( dataFrame["surv1yr"] ) )  &
#         ( pd.notnull( dataFrame["surv2yr"] ) )  &
#         ( pd.notnull( dataFrame["histology_grouped"] ) )  &
#         ( pd.notnull( dataFrame["stage"] ) ) 
#         # ( pd.notnull( dataFrame["age"] ) )  
#         ]
   
#     dataFrame = dataFrame.reset_index(drop=True)
    
#     ###### FIX ALL
    
#     #1# clean histology - remove smallcell and other
#     # histToInclude - only NSCLC
#     histToInclude = [1.0,2.0,3.0,4.0]
#     # not included - SCLC and other and no data [ 0,5,6,7,8,9 ]
#     dataFrame = dataFrame [ dataFrame.histology_grouped.isin(histToInclude) ]
#     dataFrame = dataFrame.reset_index(drop=True)

    
#     #2# use 1,2,3 stages
#     stageToInclude = [1.0,2.0,3.0]
#     dataFrame = dataFrame [ dataFrame.stage.isin(stageToInclude) ]
#     dataFrame = dataFrame.reset_index(drop=True)

        
#     ###### GET TRAINING  

#     dataFrameTrain = dataFrame [ dataFrame["dataset"].isin(trainList) ]
#     dataFrameTrain = dataFrameTrain.reset_index(drop=True)

#     # now split into training and validation
#     dataFrameTrain = dataFrameTrain.sample( frac=1 , random_state = 1 )
#     dataFrameTrain, dataFrameValidate = np.split(dataFrameTrain,[ int(.80*len(dataFrameTrain)) ])
#     dataFrameTrain = dataFrameTrain.reset_index(drop=True)
#     dataFrameValidate = dataFrameValidate.reset_index(drop=True)



    
#     ##### GET TEST
#     # for RT
#     dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
#     #3# type of treatment - use only radio or chemoRadio - use .npy file
#     chemoRadio = np.load("rt_chemoRadio.npy").astype(str)
#     dataFrameTest = dataFrameTest [ dataFrameTest["patient"].isin(chemoRadio) ]
#     #4# (rt only) use all causes of death
#     # not implemented
#     dataFrameTest = dataFrameTest.reset_index(drop=True)

#     # for lung2
#     # dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
#     # dataFrameTest = dataFrameTest.reset_index(drop=True)
    
#     print ("train patients " , dataFrameTrain.shape)
#     print ("validate patients : " , dataFrameValidate.shape) 
#     print ("test size : " , dataFrameTest.shape)

#     return dataFrameTrain, dataFrameValidate,dataFrameTest





# run 51
# def manageDataFramesRTn1():
#     trainList = ["nsclc_rt"] 
#     testList = ["lung2"] # or lung2

#     dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
#     dataFrame = dataFrame [ 
#         ( pd.notnull( dataFrame["pathToData"] ) ) &
#         ( pd.notnull( dataFrame["pathToMask"] ) ) &
#         ( pd.notnull( dataFrame["stackMin"] ) ) &
#         ( pd.isnull( dataFrame["patch_failed"] ) ) &
#         # ( pd.notnull( dataFrame["surv1yr"] ) )  &
#         ( pd.notnull( dataFrame["surv2yr"] ) )  &
#         ( pd.notnull( dataFrame["histology_grouped"] ) )  &
#         ( pd.notnull( dataFrame["stage"] ) ) 
#         # ( pd.notnull( dataFrame["age"] ) )  
#         ]
   
#     dataFrame = dataFrame.reset_index(drop=True)
    
#     ###### FIX ALL
    
#     #1# clean histology - remove smallcell and other
#     # histToInclude - only NSCLC
#     histToInclude = [1.0,2.0,3.0,4.0]
#     # not included - SCLC and other and no data [ 0,5,6,7,8,9 ]
#     dataFrame = dataFrame [ dataFrame.histology_grouped.isin(histToInclude) ]
#     dataFrame = dataFrame.reset_index(drop=True)

    
#     #2# use 1,2,3 stages
#     stageToInclude = [1.0,2.0,3.0]
#     dataFrame = dataFrame [ dataFrame.stage.isin(stageToInclude) ]
#     dataFrame = dataFrame.reset_index(drop=True)

        
#     ###### GET TRAINING  

#     dataFrameTrain = dataFrame [ dataFrame["dataset"].isin(trainList) ]
#     #3# type of treatment - use only radio or chemoRadio - use .npy file
#     chemoRadio = np.load("rt_chemoRadio.npy").astype(str)
#     dataFrameTrain = dataFrameTrain [ dataFrameTrain["patient"].isin(chemoRadio) ]
#     #4# (rt only) use all causes of death
#     # not implemented
#     dataFrameTrain = dataFrameTrain.reset_index(drop=True)

#     # now split into training and validation
#     dataFrameTrain = dataFrameTrain.sample( frac=1 , random_state = 1 )
#     dataFrameTrain, dataFrameValidate = np.split(dataFrameTrain,[ int(.87*len(dataFrameTrain)) ])
#     dataFrameTrain = dataFrameTrain.reset_index(drop=True)
#     dataFrameValidate = dataFrameValidate.reset_index(drop=True)
    
#     ##### GET TEST
#     dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
#     dataFrameTest = dataFrameTest.reset_index(drop=True)
    
#     print ("train patients " , dataFrameTrain.shape)
#     print ("validate patients : " , dataFrameValidate.shape) 
#     print ("test size : " , dataFrameTest.shape)

#     return dataFrameTrain, dataFrameValidate,dataFrameTest