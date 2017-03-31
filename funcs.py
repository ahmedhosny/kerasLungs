
from __future__ import division
from __future__ import print_function
import krs
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D , MaxPooling3D
from sklearn.metrics import roc_auc_score
import time
from keras import backend as K
import random
import tensorflow as tf
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

def manageDataFrames():
    trainList = ["nsclc_rt"]  # , , , ,  ,"oncopanel" , "moffitt","moffittSpore"  ,"oncomap" , ,"lung3" 
    validateList = ["lung1"] # leave empty
    testList = ["lung2"] # split to val and test

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

    
    #2# use 1,2,3 stages
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

    dataFrameValidate = dataFrame [ dataFrame["dataset"].isin(validateList) ]
    dataFrameValidate = dataFrameValidate.reset_index(drop=True)
    print ("validate patients : " , dataFrameValidate.shape)


    #
    # now combine train and val , then split them.
    dataFrameTrainValidate = pd.concat([dataFrameTrain,dataFrameValidate] , ignore_index=False )
    dataFrameTrainValidate = dataFrameTrainValidate.sample( frac=1 , random_state = 42 )
    dataFrameTrainValidate = dataFrameTrainValidate.reset_index(drop=True)
    print ("final - train and validate patients : " , dataFrameTrainValidate.shape)


    thirty = int(dataFrameTrainValidate.shape[0]*0.1)   ######################################
    if thirty % 2 != 0:
        thirty = thirty + 1

    # get 0's and 1's.
    zero = dataFrameTrainValidate [  (dataFrameTrainValidate['surv2yr']== 0.0)  ]
    one = dataFrameTrainValidate [  (dataFrameTrainValidate['surv2yr']== 1.0)  ]

    # split to train and val
    half = int(thirty/2.0)

    trueList = [True for i in range (half)]
    #
    zeroFalseList = [False for i in range (zero.shape[0] - half )]
    zero_msk = trueList + zeroFalseList
    random.seed(41)
    random.shuffle(zero_msk)
    zero_msk = np.array(zero_msk)
    #
    oneFalseList = [False for i in range (one.shape[0] - half )]
    one_msk = trueList + oneFalseList
    random.seed(41)
    random.shuffle(one_msk)
    one_msk = np.array(one_msk)


    # TRAIN
    zero_test = zero[~zero_msk]
    one_test = one[~one_msk]
    dataFrameTrain = pd.DataFrame()
    dataFrameTrain = dataFrameTrain.append(zero_test)
    dataFrameTrain = dataFrameTrain.append(one_test)
    dataFrameTrain = dataFrameTrain.sample( frac=1 , random_state = 42 )
    dataFrameTrain = dataFrameTrain.reset_index(drop=True)
    print ('final - train size:' , dataFrameTrain.shape)


    # VALIDATE
    zero_val = zero[zero_msk]
    one_val = one[one_msk]
    dataFrameValidate = pd.DataFrame()
    dataFrameValidate = dataFrameValidate.append(zero_val)
    dataFrameValidate = dataFrameValidate.append(one_val)
    dataFrameValidate = dataFrameValidate.sample( frac=1 , random_state = 42 )
    dataFrameValidate = dataFrameValidate.reset_index(drop=True)
    print ('final - validate size:' , dataFrameValidate.shape)


    # TEST
    dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
    dataFrameTest = dataFrameTest.reset_index(drop=True)
    print ("final - test size : " , dataFrameTest.shape)
    

    return dataFrameTrain,dataFrameValidate,dataFrameTest



def getXandY(dataFrame,imgSize):



    arrList = []
    y = []
    zeros = 0
    ones = 0
    # clincical = []
    
    for i in range (dataFrame.shape[0]):

        npy =  "/home/ubuntu/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"

        arr = np.load(npy)

        # X #
        arrList.append (  arr )  

        # Y #
        y.append ( int(dataFrame.surv2yr[i])  )

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
    y = np_utils.to_categorical(y, 2)
    # clincical = np.array(clincical , 'float32'  )
    return arrList,y,zeros,ones # ,clincical

def getX(dataFrame,imgSize):

    arrList = []

    for i in range (dataFrame.shape[0]):

        npy =  "/home/ubuntu/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"

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

    model.add(Convolution2D(32, 3, 3, border_mode='valid', dim_ordering='tf', input_shape=[imgSize,imgSize,1] , activity_regularizer = regul )) # 32
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    ##### for figure only
    # model.add(MaxPooling2D(pool_size=(3, 3))) ### 
    # model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3 , border_mode='valid', activity_regularizer = regul  )) # 32
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='valid' , activity_regularizer = regul  )) # 64
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3,  border_mode='valid' , activity_regularizer = regul )) # 64
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # # this chucnk added - 14
    # model.add(Convolution2D(256, 3, 3)) # 64
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512 , activity_regularizer = regul  )) # 512
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    return model



def make3dConvModel(imgSize,count,fork,skip):
    #(samples, rows, cols, channels) if dim_ordering='tf'.
    
    model = Sequential()

    if fork:
        model.add(Convolution3D(48, 3, 3, 3, border_mode='same',dim_ordering='tf',input_shape=[count*2+1,imgSize,imgSize,1] )) # 32
    else:
        model.add(Convolution3D(48, 3, 3, 3, border_mode='same',dim_ordering='tf',input_shape=[imgSize/skip,imgSize/skip,imgSize/skip,1] )) # 32

    model.add(BatchNormalization())
    model.add(Activation('relu'))


    model.add(Convolution3D(48, 3, 3, 3)) # 32
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3))) ### 
    model.add(Dropout(0.25))

    model.add(Convolution3D(96, 3, 3, 3, border_mode='same')) # 64
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution3D(96, 3, 3 , 3)) # 64
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512)) # 512
    model.add(BatchNormalization())
    model.add(Activation('relu'))
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

# get auc
def AUC(test_labels,test_prediction):
    n_classes = 2
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i]) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return round(roc_auc[0],3) , round(roc_auc[1],3)


class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):

        self.train_loss = []
        self.auc = []
        self.logits = []

        # save json representation
        model_json = self.model.to_json()
        with open("/home/ubuntu/output/" + RUN + "_json.json", "w") as json_file:
            json_file.write(model_json)


        dataFrameTrain,dataFrameValidate,dataFrameTest= manageDataFrames()
        #
        x_val,y_val,zeros,ones =  getXandY(dataFrameValidate,imgSize)
        print ("validation data:" , x_val.shape,  y_val.shape , zeros , ones ) 
        self.dataFrameValidate = dataFrameValidate
        self.y_val = y_val
        # lets do featurewiseCenterAndStd - its still a cube at this point
        x_val_cs = centerAndStandardizeValTest(x_val,mean,std)


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

                elif mode == "2d":
                    # get predictions
                    y_pred = self.model.predict_on_batch ( [ self.x_val[i].reshape(1,imgSize,imgSize,1) ] )

            # now after down with switching
            logits.append( y_pred[0] )



        print ( "\npredicted val zeros: "  , len( [ x for x in  logits if x[0] > x[1]  ] )  )
        print ( "predicted val ones: "  , len( [ x for x in  logits if x[0] < x[1]  ] )  )

        logits = np.array(logits)

        print ("logits: " , logits.shape , logits[0] , logits[30]   )
        auc1 , auc2 = AUC(  self.y_val ,  logits )
        print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
        print ("wtf2")

        # # before appending, check if this auc is the highest in all the list, if yes save the h5 model
        if all(auc1>i for i in self.auc):
            self.model.save_weights("/home/ubuntu/output/" + RUN + "_model.h5")
            print("Saved model to disk")
            # save model and json representation
            model_json = self.model.to_json()
            with open("/home/ubuntu/output/" + RUN + "_json.json", "w") as json_file:
                json_file.write(model_json)

        # append and save train loss
        self.train_loss.append(logs.get('loss'))
        np.save( "/home/ubuntu/output/" + RUN + "_train_loss.npy", self.train_loss) 

        # append and save auc
        self.auc.append(auc1)
        np.save( "/home/ubuntu/output/" + RUN + "_auc.npy", self.auc)

        # append and save logits
        self.logits.append(logits)
        np.save( "/home/ubuntu/output/" + RUN + "_logits.npy", self.logits)
         
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