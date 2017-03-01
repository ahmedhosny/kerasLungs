
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.metrics import roc_auc_score
import time


RUN = "2"

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
    trainList = ["lung1","lung3","oncomap" ,"oncopanel"]  # ,,"moffitt","moffittSpore","nsclc_rt"
    validateList = ["lung2"]
    testList = ["nsclc_rt"]

    dataFrame = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
    dataFrame = dataFrame [ 
    ( pd.notnull( dataFrame["pathToData"] ) ) &
    ( pd.notnull( dataFrame["pathToMask"] ) ) &
    ( pd.notnull( dataFrame["stackMin"] ) ) &
    ( pd.notnull( dataFrame["histology"] ) ) & 
    ( pd.notnull( dataFrame["surv2yr"] ) ) &
    ( pd.notnull( dataFrame["surv1yr"] ) ) & 
    ( pd.notnull( dataFrame["stage"] ) ) &
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


def getSlices2d(arr,orient,imgSize):
    if orient == "A":
        arr1 = arr[60,0:imgSize,0:imgSize]
        arr2 = arr[60,0:imgSize,30:150]
        arr3 = arr[60,30:150,30:150]
        arr4 = arr[60,30:150,0:imgSize]
        #
        arr5 = arr[75,15:135,15:135]
        #
        arr6 = arr[90,0:imgSize,0:imgSize]
        arr7 = arr[90,0:imgSize,30:150]
        arr8 = arr[90,30:150,30:150]
        arr9 = arr[90,30:150,0:imgSize]
    elif orient == "C":
        arr1 = arr[0:imgSize,60,0:imgSize]
        arr2 = arr[0:imgSize,60,30:150]
        arr3 = arr[30:150,60,30:150]
        arr4 = arr[30:150,60,0:imgSize]
        #
        arr5 = arr[15:135,75,15:135]
        #
        arr6 = arr[0:imgSize,90,0:imgSize]
        arr7 = arr[0:imgSize,90,30:150]
        arr8 = arr[30:150,90,30:150]
        arr9 = arr[30:150,90,0:imgSize]
    elif orient == "S":
        arr1 = arr[0:imgSize,0:imgSize,60]
        arr2 = arr[0:imgSize,30:150,60]
        arr3 = arr[30:150,30:150,60]
        arr4 = arr[30:150,0:imgSize,60]
        #
        arr5 = arr[15:135,15:135,75]
        #
        arr6 = arr[0:imgSize,0:imgSize,90]
        arr7 = arr[0:imgSize,30:150,90]
        arr8 = arr[30:150,30:150,90]
        arr9 = arr[30:150,0:imgSize,90]
        
    return  [arr1.reshape(imgSize,imgSize,1) 
            ,arr2.reshape(imgSize,imgSize,1) 
            ,arr3.reshape(imgSize,imgSize,1) 
            ,arr4.reshape(imgSize,imgSize,1) 
            ,arr5.reshape(imgSize,imgSize,1)
            ,arr6.reshape(imgSize,imgSize,1) 
            ,arr7.reshape(imgSize,imgSize,1) 
            ,arr8.reshape(imgSize,imgSize,1) 
            ,arr9.reshape(imgSize,imgSize,1)]

def getSlices3d(arr,orient,imgSize):
    # current: 5 slices each 4mm/pixel apart
    # number of slices in each direction
    count = 2
    # skip how many slices - every other one
    skip = 4
    # travel
    travel = count * skip
    # always an odd number of slices around the center slice
    if orient == "A": 
        arr1 = arr[(60-travel):(60+travel+1):skip,0:imgSize,0:imgSize]
        print [(60-travel),(60+travel+1),skip,0,imgSize,0,imgSize]
        arr2 = arr[(60-travel):(60+travel+1):skip,0:imgSize,30:150]
        arr3 = arr[(60-travel):(60+travel+1):skip,30:150,30:150]
        arr4 = arr[(60-travel):(60+travel+1):skip,30:150,0:imgSize]
        #
        arr5 = arr[(75-travel):(75+travel+1):skip,15:135,15:135]
        #
        arr6 = arr[(90-travel):(90+travel+1):skip,0:imgSize,0:imgSize]
        arr7 = arr[(90-travel):(90+travel+1):skip,0:imgSize,30:150]
        arr8 = arr[(90-travel):(90+travel+1):skip,30:150,30:150]
        arr9 = arr[(90-travel):(90+travel+1):skip,30:150,0:imgSize]
    elif orient == "C":
        arr1 = arr[0:imgSize,(60-travel):(60+travel+1):skip,0:imgSize]
        arr2 = arr[0:imgSize,(60-travel):(60+travel+1):skip,30:150]
        arr3 = arr[30:150,(60-travel):(60+travel+1):skip,30:150]
        arr4 = arr[30:150,(60-travel):(60+travel+1):skip,0:imgSize]
        #
        arr5 = arr[15:135,(75-travel):(75+travel+1):skip,15:135]
        #
        arr6 = arr[0:imgSize,(90-travel):(90+travel+1):skip,0:imgSize]
        arr7 = arr[0:imgSize,(90-travel):(90+travel+1):skip,30:150]
        arr8 = arr[30:150,(90-travel):(90+travel+1):skip,30:150]
        arr9 = arr[30:150,(90-travel):(90+travel+1):skip,0:imgSize]
    elif orient == "S":
        arr1 = arr[0:imgSize,0:imgSize,(60-travel):(60+travel+1):skip]
        arr2 = arr[0:imgSize,30:150,(60-travel):(60+travel+1):skip]
        arr3 = arr[30:150,30:150,(60-travel):(60+travel+1):skip]
        arr4 = arr[30:150,0:imgSize,(60-travel):(60+travel+1):skip]
        #
        arr5 = arr[15:135,15:135,(75-travel):(75+travel+1):skip]
        #
        arr6 = arr[0:imgSize,0:imgSize,(90-travel):(90+travel+1):skip]
        arr7 = arr[0:imgSize,30:150,(90-travel):(90+travel+1):skip]
        arr8 = arr[30:150,30:150,(90-travel):(90+travel+1):skip]
        arr9 = arr[30:150,0:imgSize,(90-travel):(90+travel+1):skip]
        
    return  [arr1.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr2.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr3.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr4.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr5.reshape(count*2+1,imgSize,imgSize,1)
            ,arr6.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr7.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr8.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr9.reshape(count*2+1,imgSize,imgSize,1)]


def getXandY(dataFrame):

    imgSize = 120

    a = []
    s = []
    c = []
    y = []
    zeros = 0
    ones = 0
    
    for i in range (dataFrame.shape[0]):

        npy =  "/home/ubuntu/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"

        arr = np.load(npy)
        # X #
        a.extend (  getSlices2d(arr,'A',imgSize) ) # adds 9 images   ####################################################################################################################
        s.extend (  getSlices2d(arr,'S',imgSize) ) # adds 9 images   ###################################################################################################################
        c.extend (  getSlices2d(arr,'C',imgSize) ) # adds 9 images   ###################################################################################################################
        # Y #
        y.extend ( [ int(dataFrame.surv2yr[i]) for x in range (9) ] )
        # zeros and ones
        if int(dataFrame.surv2yr[i]) == 1:
            ones = ones+1
        elif int(dataFrame.surv2yr[i]) == 0:
            zeros = zeros+1
        else:
            raise Exception("a survival value is not 0 or 1")
            
    # after loop
    a = np.array(a, 'float32')
    s = np.array(s, 'float32')
    c = np.array(c, 'float32')
    y = np.array(y, 'int8')
    y = np_utils.to_categorical(y, 2)
    
    return a,s,c,y,zeros,ones

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

def make2dConvModel():
    #(samples, rows, cols, channels) if dim_ordering='tf'.
    
    model = Sequential()

    model.add(Convolution2D(48, 5, 5, border_mode='same',dim_ordering='tf',input_shape=[120,120,1] )) # 32
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

    model.add(Flatten())
    model.add(Dense(512)) # 512
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    return model

def make3dConvModel():
    # (samples, conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'.

    model = Sequential()

    model.add(Convolution3D(32, 5, 5, 5, border_mode='same',dim_ordering='tf' ,input_shape=[5,120,120,1] )) # 32
    model.add(Activation('relu'))

    model.add(Convolution3D(32, 5, 5, 5)) # 32
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution3D(64, 5, 5, 5, border_mode='same')) # 64
    model.add(Activation('relu'))
    model.add(Convolution3D(64, 5, 5, 5)) # 64
    model.add(Activation('relu'))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512)) # 512
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


def featurewiseCenterAndStd(arr):

    myMean = np.mean(arr, axis=(0, 1, 2))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[3 - 1] = arr.shape[3]
    myMean = np.reshape(myMean, broadcast_shape)
    arr -= myMean

    mySTD = np.std(arr, axis=(0, 1, 2))
    broadcast_shape = [1, 1, 1]
    broadcast_shape[3 - 1] = arr.shape[3]
    mySTD = np.reshape(mySTD, broadcast_shape)
    arr /= ( mySTD + np.finfo(float).eps )

    return arr

# used for evaluating performance 
def aggregate(testLabels,logits):
    mul = 9 # every 9
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
        self.validation_logits = []
        self.train_loss = []
        self.validation_loss = []
        self.count = 7
        self.reduced = 135

        dataFrameTrain,dataFrameValidate,dataFrameTest= manageDataFrames()
        #
        x_test_a , x_test_s , x_test_c , y_test, zeros , ones = getXandY(dataFrameValidate)
        print ("test data: " ,x_test_a.shape , x_test_s.shape , x_test_c.shape , y_test.shape )


        self.y_test = getChuncks(y_test, self.count , self.reduced)
        self.x_test_a = getChuncks(x_test_a, self.count , self.reduced)   
        self.x_test_s = getChuncks(x_test_s, self.count , self.reduced)
        self.x_test_c = getChuncks(x_test_c, self.count , self.reduced)

        print ("part of test data: " , self.x_test_a[0].shape )
        print ("part of test labels: " , self.y_test[0].shape )


    def on_train_end(self, logs={}):

        # save model and json representation
        model_json = self.model.to_json()
        with open("/home/ubuntu/output/" + RUN + "_json.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("/home/ubuntu/output/" + RUN + "_model.h5")
        print("Saved model to disk")

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
        #
        for i in range (self.count ):
            # get predictions
            y_pred = self.model.predict_on_batch ( [ self.x_test_a[i] , self.x_test_s[i] , self.x_test_c[i] ]  )
            # group by patient - to get one prediction per patient only
            labelsOut,logitsOut = aggregate( self.y_test[i] , y_pred )
            #
            allLabels.extend(labelsOut)
            allLogits.extend(logitsOut)
            #

        allLabels = np.array(allLabels)
        allLogits = np.array(allLogits)
        # 
        print ("\nfinal labels,logits shape: " , allLabels.shape , allLogits.shape )


        # get 2 auc's
        print ("wtf1")
        auc1 , auc2 = AUC(  allLabels ,  allLogits )
        print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
        self.auc.append(auc1)
        print ("wtf2")



        self.validation_logits.append(allLogits)
        self.train_loss.append(logs.get('loss'))
        

        # overwrite every time - no problem
        # save stuff
        np.save( "/home/ubuntu/output/" + RUN + "_auc.npy", self.auc)
        np.save( "/home/ubuntu/output/" + RUN + "_validation_logits.npy", self.validation_logits)
        np.save( "/home/ubuntu/output/" + RUN + "_train_loss.npy", self.train_loss)


        # test loss
        #
        for i in range (self.count ):
            # now do loss
            temp = self.model.test_on_batch ( [ self.x_test_a[i] , self.x_test_s[i] , self.x_test_c[i] ]  , self.y_test[i] )
            validation_loss.append ( temp )
        validation_loss_avg = np.mean(validation_loss)
        self.validation_loss.append(validation_loss_avg)
        np.save( "/home/ubuntu/output/" + RUN + "_validation_loss.npy", self.validation_loss)
         
        
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):

        return


#
#
#     `7MMF' `YMM' `7MM"""YMM  `7MM"""Mq.        db       .M"""bgd       .g8"""bgd `7MM"""YMM  `7MN.   `7MF'`7MM"""YMM  `7MM"""Mq.        db   MMP""MM""YMM   .g8""8q. `7MM"""Mq.
#       MM   .M'     MM    `7    MM   `MM.      ;MM:     ,MI    "Y     .dP'     `M   MM    `7    MMN.    M    MM    `7    MM   `MM.      ;MM:  P'   MM   `7 .dP'    `YM. MM   `MM.
#       MM .d"       MM   d      MM   ,M9      ,V^MM.    `MMb.         dM'       `   MM   d      M YMb   M    MM   d      MM   ,M9      ,V^MM.      MM      dM'      `MM MM   ,M9
#       MMMMM.       MMmmMM      MMmmdM9      ,M  `MM      `YMMNq.     MM            MMmmMM      M  `MN. M    MMmmMM      MMmmdM9      ,M  `MM      MM      MM        MM MMmmdM9
#       MM  VMA      MM   Y  ,   MM  YM.      AbmmmqMA   .     `MM     MM.    `7MMF' MM   Y  ,   M   `MM.M    MM   Y  ,   MM  YM.      AbmmmqMA     MM      MM.      ,MP MM  YM.
#       MM   `MM.    MM     ,M   MM   `Mb.   A'     VML  Mb     dM     `Mb.     MM   MM     ,M   M     YMM    MM     ,M   MM   `Mb.   A'     VML    MM      `Mb.    ,dP' MM   `Mb.
#     .JMML.   MMb..JMMmmmmMMM .JMML. .JMM..AMA.   .AMMA.P"Ybmmd"        `"bmmmdPY .JMMmmmmMMM .JML.    YM  .JMMmmmmMMM .JMML. .JMM..AMA.   .AMMA..JMML.      `"bmmd"' .JMML. .JMM.
#
# 


def createGenerator( A, S, C, Y, batch_size, generator):

    while True:
        # suffled indices    
        idx = np.random.permutation( A.shape[0])
        # create image generator
        batches_A = generator.flow( A[idx], Y[idx], batch_size=batch_size , shuffle=False)
        batches_S = generator.flow( S[idx], Y[idx], batch_size=batch_size , shuffle=False)
        batches_C = generator.flow( C[idx], Y[idx], batch_size=batch_size , shuffle=False)

        for batch_a , batch_s , batch_c , counter in zip( 
            batches_A , batches_S , batches_C , np.arange(batch_size) ):

            yield [ batch_a[0], batch_s[0] , batch_c[0] ] , batch_a[1]

            if counter >= batch_size:
                break




