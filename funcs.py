
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
    validateList = ["lung2"]
    testList = ["lung1"]

    dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
    dataFrame = dataFrame [ 
    ( pd.notnull( dataFrame["pathToData"] ) ) &
    ( pd.notnull( dataFrame["pathToMask"] ) ) &
    ( pd.notnull( dataFrame["stackMin"] ) ) &
    ( pd.notnull( dataFrame["histology"] ) ) & 
    ( pd.notnull( dataFrame["surv2yr"] ) ) &
    ( pd.notnull( dataFrame["surv1yr"] ) ) & 
    ( pd.notnull( dataFrame["stage"] ) ) &
    ( pd.notnull( dataFrame["age"] ) ) &
    ( pd.isnull( dataFrame["patch_failed"] ) )
    # & ( dataFrame["stage"] == 1.0 ) 
    ]
    dataFrame = dataFrame.reset_index(drop=True)
    print ("all patients: " , dataFrame.shape)
    #
    dataFrameTrain = dataFrame [ dataFrame["dataset"].isin(trainList) ]
    dataFrameTrain = dataFrameTrain.reset_index(drop=True)
    print ("train patients: " , dataFrameTrain.shape)
    #
    dataFrameValidate = dataFrame [ dataFrame["dataset"].isin(validateList) ]
    dataFrameValidate = dataFrameValidate.reset_index(drop=True)
    print ("validate patients: " , dataFrameValidate.shape)
    #
    dataFrameTest = dataFrame [ dataFrame["dataset"].isin(testList) ]
    dataFrameTest = dataFrameTest.reset_index(drop=True)
    print ("test patients: " , dataFrameTest.shape)

    return dataFrameTrain,dataFrameValidate,dataFrameTest



def getXandY(dataFrame,imgSize, bool):

    # if train, aug factor is always one.
    augmentationFactor = 1
    
    # if validate or test, change if needed
    if (bool):
        augmentationFactor = valTestMultiplier


    arrList = []
    y = []
    zeros = 0
    ones = 0
    clincical = []
    
    for i in range (dataFrame.shape[0]):

        npy =  "/home/ubuntu/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"

        arr = np.load(npy)

        # X #
        arrList.append (  arr )  

        # Y #
        y.extend ( [ int(dataFrame.surv2yr[i]) for x in range (augmentationFactor) ] )

        # zeros and ones
        if int(dataFrame.surv2yr[i]) == 1:
            ones = ones+1
        elif int(dataFrame.surv2yr[i]) == 0:
            zeros = zeros+1
        else:
            raise Exception("a survival value is not 0 or 1")

        # now clinical
        clincicalVector = [ dataFrame.age[i] , dataFrame.stage[i] , dataFrame.histology_grouped[i] ]
        clincical.extend( [clincicalVector for x in range(augmentationFactor)] )


    # after loop
    arrList = np.array(arrList, 'float32')
    y = np.array(y, 'int8')
    y = np_utils.to_categorical(y, 2)
    clincical = np.array(clincical , 'float32'  )
    return arrList,y,zeros,ones,clincical



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
def makeClinicalModel():
    model = Sequential()
    # just histology, stage and age
    model.add(Dense( 3, input_dim=(3)) ) # 512
    return model

def make2dConvModel(imgSize):
    #(samples, rows, cols, channels) if dim_ordering='tf'.
    
    model = Sequential()

    model.add(Convolution2D(48, 5, 5, border_mode='same',dim_ordering='tf',input_shape=[imgSize,imgSize,1] )) # 32
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 5, 5)) # 32
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 5, 5, border_mode='same')) # 64
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 5, 5)) # 64
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # this chucnk added - 14
    model.add(Convolution2D(192, 5, 5)) # 64
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512)) # 512
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    return model


def make3dConvModel(imgSize,count):
    #(samples, rows, cols, channels) if dim_ordering='tf'.
    
    model = Sequential()

    model.add(Convolution3D(48, 2, 5, 5, border_mode='same',dim_ordering='tf',input_shape=[count*2+1,imgSize,imgSize,1] )) # 32
    model.add(Activation('relu'))

    model.add(Convolution3D(48, 2, 5, 5)) # 32
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2))) ### 
    model.add(Dropout(0.25))

    model.add(Convolution3D(96, 2, 5, 5, border_mode='same')) # 64
    model.add(Activation('relu'))

    model.add(Convolution3D(96, 1, 5 , 5)) # 64
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(512)) # 512
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    return model


# def make3dConvModel(imgSize,count):
#     # (samples, conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'.

#     model = Sequential()

#     conv_filt = 3
#     conv_filt_depth = 2
#     #

#     # input = (samples, count*2+1,imgSize,imgSize,1 )
#     model.add(Convolution3D(48, conv_filt_depth , conv_filt, conv_filt, border_mode='same',dim_ordering='tf' ,input_shape=[count*2+1,imgSize,imgSize,1]  , activation='relu')) # 32
#     # output (samples, count*2+1,imgSize,imgSize, nb_filter)

#     model.add( Convolution3D( 48, conv_filt_depth , conv_filt, conv_filt, border_mode='same' , activation='relu' , dim_ordering='tf'  ) ) # 32
#     model.add( MaxPooling3D( pool_size=(3, 2, 2) , dim_ordering='tf'  ) )
#     model.add( Dropout(0.5) )

#     model.add(Convolution3D(96, conv_filt_depth , conv_filt, conv_filt,  border_mode='same' , activation='relu'   , dim_ordering='tf' )) # 64

#     model.add(Convolution3D(96, conv_filt_depth , conv_filt, conv_filt,  border_mode='same' , activation='relu'  , dim_ordering='tf'  )) # 64
#     model.add(MaxPooling3D(pool_size=(3, 2, 2) , dim_ordering='tf' ))
#     model.add(Dropout(0.5))
    
#     # model.add(Convolution3D(192, conv_filt_depth , conv_filt, conv_filt,  border_mode='same' , activation='relu'  , dim_ordering='tf'  )) # 64   

#     model.add(Flatten())
#     model.add( Dense(512 , activation='relu' ) ) # 512
#     model.add(Dropout(0.5))
    
#     return model


def makeSingle3dConvModel(imgSize,skip):
    # (samples, conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'.

    model = Sequential()

    conv_filt = 3
    conv_filt_depth = 2
    #

    # input = (samples, count*2+1,imgSize,imgSize,1 )
    model.add(Convolution3D(48, conv_filt_depth , conv_filt, conv_filt, border_mode='same',dim_ordering='tf' ,input_shape=[imgSize/skip,imgSize/skip,imgSize/skip,1]  , activation='relu')) # 32
    # output (samples, count*2+1,imgSize,imgSize, nb_filter)

    model.add( Convolution3D( 48, conv_filt_depth , conv_filt, conv_filt, border_mode='same' , activation='relu' , dim_ordering='tf'  ) ) # 32
    model.add( MaxPooling3D( pool_size=(3, 2, 2) , dim_ordering='tf'  ) )
    model.add( Dropout(0.5) )

    model.add(Convolution3D(96, conv_filt_depth , conv_filt, conv_filt,  border_mode='same' , activation='relu'   , dim_ordering='tf' )) # 64

    model.add(Convolution3D(96, conv_filt_depth , conv_filt, conv_filt,  border_mode='same' , activation='relu'  , dim_ordering='tf'  )) # 64
    model.add(MaxPooling3D(pool_size=(3, 2, 2) , dim_ordering='tf' ))
    model.add(Dropout(0.5))
    
    # model.add(Convolution3D(192, conv_filt_depth , conv_filt, conv_filt,  border_mode='same' , activation='relu'  , dim_ordering='tf'  )) # 64   

    model.add(Flatten())
    model.add( Dense(512 , activation='relu' ) ) # 512
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

# NEW
# operate on each orientation seperately

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

def centerAndStandardizeValTest(arr,mean,std):
    out = arr
    #
    out -= mean
    out /= (std + np.finfo(float).eps )
    #
    return out

# OLD


# used for evaluating performance 
def aggregate(testLabels,logits):
    mul = valTestMultiplier
    labelsOut = []
    logitsOut = []
    # 
    for i in range ( 0,testLabels.shape[0],mul ):
        labelsOut.append( testLabels[i] )

    #
    for i in range ( 0,testLabels.shape[0],mul ):
        tempVal0 = 0
        tempVal1 = 0
        for k in range (mul):
            tempVal0 += logits[i+k][0]
            tempVal1 += logits[i+k][1]
        val0 = tempVal0 / (mul*1.0)
        val1 = tempVal1 / (mul*1.0)
        logitsOut.append( [ val0,val1 ] )
    #
    return np.array(labelsOut),np.array(logitsOut)


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

# returns a list of divided chuncks from the given param
# reduced must be divisible by 9
def getChuncks(inputa, count, reduced):

    output = []

    # will handle all but last one
    for i in range (count-1):
        output.append( inputa [ reduced*i : reduced*(i+1) ] ) 
    # will handle last one
    output.append( inputa [ reduced*(count-1) : ] ) 

    return output

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.auc = []
        self.val_logits = []
        self.val_logits_raw = []
        self.train_loss = []
        self.val_loss = []
        self.count = 3 #21 ##############################################################################################################
        self.reduced = 45

        dataFrameTrain,dataFrameValidate,dataFrameTest= manageDataFrames()
        #
        x_val,y_val,zeros,ones,clinical_val =  getXandY(dataFrameValidate,imgSize, True)
        print ("validation data:" , x_val.shape,  y_val.shape , clinical_val.shape ) 

        # lets do featurewiseCenterAndStd - its still a cube at this point
        x_val_cs = centerAndStandardizeValTest(x_val,mean,std)


        if fork:
            # lets get the 3 orientations
            x_val_a,x_val_s,x_val_c = krs.splitValTest(x_val_cs,finalSize,imgSize,count,mode)
            print ("final val data:" , x_val_a.shape,x_val_s.shape,x_val_c.shape)

            # now lets break them into chuncks divisible by 9 to fit into the GPU
            self.y_val = getChuncks(y_val, self.count , self.reduced)
            self.x_val_a = getChuncks(x_val_a, self.count , self.reduced)   
            self.x_val_s = getChuncks(x_val_s, self.count , self.reduced)
            self.x_val_c = getChuncks(x_val_c, self.count , self.reduced)
            self.clinical_val = getChuncks(clinical_val, self.count , self.reduced)

            print ("part of validate data: " , self.x_val_a[0].shape )
            print ("part of validate labels: " , self.y_val[0].shape )

        else:
            x_val = krs.splitValTest_single3D(x_val_cs,finalSize,imgSize,skip)

            print ("final val data:" , x_val.shape)

            # now lets break them into chuncks divisible by 9 to fit into the GPU
            self.y_val = getChuncks(y_val, self.count , self.reduced)
            self.x_val = getChuncks(x_val, self.count , self.reduced)   
            self.clinical_val = getChuncks(clinical_val, self.count , self.reduced)

            print ("part of validate data: " , self.x_val[0].shape )
            print ("part of validate labels: " , self.y_val[0].shape )


    def on_train_end(self, logs={}):

        # save model and json representation
        model_json = self.model.to_json()
        with open("/home/ubuntu/output/" + RUN + "_json.json", "w") as json_file:
            json_file.write(model_json)

        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        #
        # PREDICT
        #
        allLabels = []
        allLogits = []
        validation_loss = []
        rawLogits = []
        #
        for i in range (self.count ):

            if fork:
                # get predictions
                y_pred = self.model.predict_on_batch ( [ self.clinical_val[i] , self.x_val_a[i] , self.x_val_s[i] , self.x_val_c[i] ]  )
            else:
                y_pred = self.model.predict_on_batch ( [ self.clinical_val[i] , self.x_val[i] ]  )

            # save raw logits
            rawLogits.extend( y_pred  ) 
            # group by patient - to get one prediction per patient only
            labelsOut,logitsOut = aggregate( self.y_val[i] , y_pred )
            #
            allLabels.extend(labelsOut)
            allLogits.extend(logitsOut)
            #

        self.val_logits_raw.append( rawLogits )
        np.save( "/home/ubuntu/output/" + RUN + "_validation_logits_raw.npy", self.val_logits_raw)


        allLabels = np.array(allLabels)
        allLogits = np.array(allLogits)

        # 
        print ("\nfinal labels,logits shape: " , allLabels.shape , allLogits.shape )


        # get 2 auc's
        print ("wtf1")
        auc1 , auc2 = AUC(  allLabels ,  allLogits )
        print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
        # before appending, check if this auc is the highest in all the lsit

        if all(auc1>i for i in self.auc):
            self.model.save_weights("/home/ubuntu/output/" + RUN + "_model.h5")
            print("Saved model to disk")

        self.auc.append(auc1)
        print ("wtf2")


        self.val_logits.append(allLogits)
        self.train_loss.append(logs.get('loss'))
        

        # overwrite every time - no problem
        # save stuff
        np.save( "/home/ubuntu/output/" + RUN + "_auc.npy", self.auc)
        np.save( "/home/ubuntu/output/" + RUN + "_validation_logits.npy", self.val_logits)
        np.save( "/home/ubuntu/output/" + RUN + "_train_loss.npy", self.train_loss)


        # validate loss
        #
        for i in range (self.count ):
            # now do loss

            if fork:
                temp = self.model.test_on_batch ( [ self.clinical_val[i] , self.x_val_a[i] , self.x_val_s[i] , self.x_val_c[i] ]  , self.y_val[i] )
            else:
                temp = self.model.test_on_batch ( [ self.clinical_val[i] , self.x_val[i] ]  , self.y_val[i] )

            validation_loss.append ( temp )
        validation_loss_avg = np.mean(validation_loss)
        self.val_loss.append(validation_loss_avg)
        np.save( "/home/ubuntu/output/" + RUN + "_validation_loss.npy", self.val_loss)
         
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):

        return

