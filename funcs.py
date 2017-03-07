
from __future__ import division
from __future__ import print_function
# import script
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
    trainList = ["lung1","lung3","oncomap" ]  # , ,"oncopanel" , "moffitt","moffittSpore"
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


def getSlices2d(arr,orient,imgSize):
    lower = 150-imgSize
    mid1 = int(lower/2.0)
    mid2 = 150-int(lower/2.0)

    if orient == "A":
        arr1 = arr[60,0:imgSize,0:imgSize]
        arr2 = arr[60,0:imgSize,lower:150]
        arr3 = arr[60,lower:150,lower:150]
        arr4 = arr[60,lower:150,0:imgSize]
        #
        arr5 = arr[75, mid1:mid2 , mid1:mid2 ]
        #
        arr6 = arr[90,0:imgSize,0:imgSize]
        arr7 = arr[90,0:imgSize,lower:150]
        arr8 = arr[90,lower:150,lower:150]
        arr9 = arr[90,lower:150,0:imgSize]

    elif orient == "C":
        arr1 = arr[0:imgSize,60,0:imgSize]
        arr2 = arr[0:imgSize,60,lower:150]
        arr3 = arr[lower:150,60,lower:150]
        arr4 = arr[lower:150,60,0:imgSize]
        #
        arr5 = arr[mid1:mid2,75,mid1:mid2]
        #
        arr6 = arr[0:imgSize,90,0:imgSize]
        arr7 = arr[0:imgSize,90,lower:150]
        arr8 = arr[lower:150,90,lower:150]
        arr9 = arr[lower:150,90,0:imgSize]

    elif orient == "S":
        arr1 = arr[0:imgSize,0:imgSize,60]
        arr2 = arr[0:imgSize,lower:150,60]
        arr3 = arr[lower:150,lower:150,60]
        arr4 = arr[lower:150,0:imgSize,60]
        #
        arr5 = arr[mid1:mid2,mid1:mid2,75]
        #
        arr6 = arr[0:imgSize,0:imgSize,90]
        arr7 = arr[0:imgSize,lower:150,90]
        arr8 = arr[lower:150,lower:150,90]
        arr9 = arr[lower:150,0:imgSize,90]
        
    return  [arr1.reshape(imgSize,imgSize,1) 
            ,arr2.reshape(imgSize,imgSize,1) 
            ,arr3.reshape(imgSize,imgSize,1) 
            ,arr4.reshape(imgSize,imgSize,1) 
            ,arr5.reshape(imgSize,imgSize,1)
            ,arr6.reshape(imgSize,imgSize,1) 
            ,arr7.reshape(imgSize,imgSize,1) 
            ,arr8.reshape(imgSize,imgSize,1) 
            ,arr9.reshape(imgSize,imgSize,1)]

def getSlices3d(arr,orient,imgSize,count):
    # var
    lower = 150-imgSize
    mid1 = int(lower/2.0)
    mid2 = 150-int(lower/2.0)

    # current: 5 slices each 4mm/pixel apart
    # count = number of slices in each direction
    # skip how many slices - every other one
    skip = 4
    # travel
    travel = count * skip
    # always an odd number of slices around the center slice
    if orient == "A": 
        arr1 = arr[(60-travel):(60+travel+1):skip,0:imgSize,0:imgSize]
        arr2 = arr[(60-travel):(60+travel+1):skip,0:imgSize,lower:150]
        arr3 = arr[(60-travel):(60+travel+1):skip,lower:150,lower:150]
        arr4 = arr[(60-travel):(60+travel+1):skip,lower:150,0:imgSize]
        #
        arr5 = arr[(75-travel):(75+travel+1):skip,mid1:mid2,mid1:mid2]
        #
        arr6 = arr[(90-travel):(90+travel+1):skip,0:imgSize,0:imgSize]
        arr7 = arr[(90-travel):(90+travel+1):skip,0:imgSize,lower:150]
        arr8 = arr[(90-travel):(90+travel+1):skip,lower:150,lower:150]
        arr9 = arr[(90-travel):(90+travel+1):skip,lower:150,0:imgSize]

    elif orient == "C":
        arr1 = arr[0:imgSize,(60-travel):(60+travel+1):skip,0:imgSize]
        arr2 = arr[0:imgSize,(60-travel):(60+travel+1):skip,lower:150]
        arr3 = arr[lower:150,(60-travel):(60+travel+1):skip,lower:150]
        arr4 = arr[lower:150,(60-travel):(60+travel+1):skip,0:imgSize]
        #
        arr5 = arr[mid1:mid2,(75-travel):(75+travel+1):skip,mid1:mid2]
        #
        arr6 = arr[0:imgSize,(90-travel):(90+travel+1):skip,0:imgSize]
        arr7 = arr[0:imgSize,(90-travel):(90+travel+1):skip,lower:150]
        arr8 = arr[lower:150,(90-travel):(90+travel+1):skip,lower:150]
        arr9 = arr[lower:150,(90-travel):(90+travel+1):skip,0:imgSize]

    elif orient == "S":
        arr1 = arr[0:imgSize,0:imgSize,(60-travel):(60+travel+1):skip]
        arr2 = arr[0:imgSize,lower:150,(60-travel):(60+travel+1):skip]
        arr3 = arr[lower:150,lower:150,(60-travel):(60+travel+1):skip]
        arr4 = arr[lower:150,0:imgSize,(60-travel):(60+travel+1):skip]
        #
        arr5 = arr[mid1:mid2,mid1:mid2,(75-travel):(75+travel+1):skip]
        #
        arr6 = arr[0:imgSize,0:imgSize,(90-travel):(90+travel+1):skip]
        arr7 = arr[0:imgSize,lower:150,(90-travel):(90+travel+1):skip]
        arr8 = arr[lower:150,lower:150,(90-travel):(90+travel+1):skip]
        arr9 = arr[lower:150,0:imgSize,(90-travel):(90+travel+1):skip]
        
    return  [arr1.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr2.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr3.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr4.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr5.reshape(count*2+1,imgSize,imgSize,1)
            ,arr6.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr7.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr8.reshape(count*2+1,imgSize,imgSize,1) 
            ,arr9.reshape(count*2+1,imgSize,imgSize,1)]


def getXandY(dataFrame,mode,imgSize,count, bool):

    _augmentationFactor = augmentationFactor
    # if validate or test
    if (bool):
        _augmentationFactor = 9

    a = []
    s = []
    c = []
    y = []
    zeros = 0
    ones = 0
    clincical = []
    
    for i in range (dataFrame.shape[0]):

        npy =  "/home/ubuntu/data/" + str(dataFrame.dataset[i]) + "_" + str(dataFrame.patient[i]) + ".npy"

        arr = np.load(npy)
        # X #
        if mode == "3d":
            a.extend (  getSlices3d(arr,'A',imgSize,count) ) # adds 9 images   
            s.extend (  getSlices3d(arr,'S',imgSize,count) ) # adds 9 images   
            c.extend (  getSlices3d(arr,'C',imgSize,count) ) # adds 9 images   
        elif mode == "2d" :
            a.extend (  getSlices2d(arr,'A',imgSize) ) # adds 9 images  
            s.extend (  getSlices2d(arr,'S',imgSize) ) # adds 9 images   
            c.extend (  getSlices2d(arr,'C',imgSize) ) # adds 9 images   

        # Y #
        y.extend ( [ int(dataFrame.surv2yr[i]) for x in range (_augmentationFactor) ] )
        # zeros and ones
        if int(dataFrame.surv2yr[i]) == 1:
            ones = ones+1
        elif int(dataFrame.surv2yr[i]) == 0:
            zeros = zeros+1
        else:
            raise Exception("a survival value is not 0 or 1")

        # now clinical
        clincicalVector = [ dataFrame.age[i] , dataFrame.stage[i] , dataFrame.histology_grouped[i] ]
        clincical.extend( [clincicalVector for x in range(_augmentationFactor)] )


    # after loop
    a = np.array(a, 'float32')
    s = np.array(s, 'float32')
    c = np.array(c, 'float32')
    y = np.array(y, 'int8')
    y = np_utils.to_categorical(y, 2)
    clincical = np.array(clincical , 'float32'  )
    return a,s,c,y,zeros,ones,clincical


# augment
# flip all direction
def flipAllThreeDirections(arr,mode):
    flip_ud = np.fliplr(arr) 
    flip_io = np.flipud(arr)
    if mode == "3d": 
        flip_rl = arr[:,:,::-1]
        return [flip_ud , flip_io , flip_rl]
    elif mode == "2d": 
        return [flip_ud , flip_io]
    
# adds original, then adds augmented - changes structure (doesnt append at end)
def augmentTraining(arr_a,arr_s,arr_c,mode):
    out_a = []
    out_s = []
    out_c = []
    # loop
    for k in range (arr_a.shape[0]):
        # Axial
        # append original
        out_a.append( arr_a[k] )
        # extend augemented
        out_a.extend(  flipAllThreeDirections(arr_a[k],mode)  )
        
        # Sagittal
        # append original
        out_s.append( arr_s[k] )
        # extend augemented
        out_s.extend(  flipAllThreeDirections(arr_s[k],mode)  )
        
        # Coronal
        # append original
        out_c.append( arr_c[k] )
        # extend augemented
        out_c.extend(  flipAllThreeDirections(arr_c[k],mode)  )
        
        
    return np.array (out_a , 'float32' )  , np.array (out_s , 'float32' )  , np.array (out_c , 'float32' ) 


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

    model.add(Flatten())
    model.add(Dense(512)) # 512
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    return model


def make3dConvModel(imgSize,count):
    # (samples, conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'.

    model = Sequential()

    # input = (samples, count*2+1,imgSize,imgSize,1 )
    model.add(Convolution3D(32, 5, 5, 5, border_mode='same',dim_ordering='tf' ,input_shape=[count*2+1,imgSize,imgSize,1]  , activation='relu')) # 32
    # output (samples, count*2+1,imgSize,imgSize, nb_filter)

    model.add( Convolution3D( 32, 5, 5, 5, border_mode='same' , activation='relu' , dim_ordering='tf'  ) ) # 32
    model.add( MaxPooling3D( pool_size=(2, 2, 2) , dim_ordering='tf'  ) )
    model.add( Dropout(0.25) )

    model.add(Convolution3D(64, 5, 5, 5,  border_mode='same' , activation='relu'   , dim_ordering='tf' )) # 64

    model.add(Convolution3D(64, 5, 5, 5 ,  border_mode='same' , activation='relu'  , dim_ordering='tf'  )) # 64
    model.add(MaxPooling3D(pool_size=(2, 2, 2) , dim_ordering='tf' ))
    model.add(Dropout(0.25))

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
        self.count = 21
        self.reduced = 45

        dataFrameTrain,dataFrameValidate,dataFrameTest= manageDataFrames()
        #
        x_validate_a , x_validate_s , x_validate_c , y_validate, zeros , ones , clincial = getXandY(dataFrameValidate, mode, imgSize, count, True)
        print ("validation data: " ,x_validate_a.shape , x_validate_s.shape , x_validate_c.shape , y_validate.shape, clincial.shape )

        # lets do featurewiseCenterAndStd
        x_validate_a = centerAndStandardizeValTest(x_validate_a,mean_a,std_a)
        x_validate_s = centerAndStandardizeValTest(x_validate_s,mean_s,std_s)
        x_validate_c = centerAndStandardizeValTest(x_validate_c,mean_c,std_c)     

        # now lets break them into chuncks divisible by 9 to fit into the GPU
        self.y_validate = getChuncks(y_validate, self.count , self.reduced)
        self.x_validate_a = getChuncks(x_validate_a, self.count , self.reduced)   
        self.x_validate_s = getChuncks(x_validate_s, self.count , self.reduced)
        self.x_validate_c = getChuncks(x_validate_c, self.count , self.reduced)
        self.clinical = getChuncks(clincial, self.count , self.reduced)

        print ("part of validate data: " , self.x_validate_a[0].shape )
        print ("part of validate labels: " , self.y_validate[0].shape )


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
        #
        for i in range (self.count ):
            # get predictions
            y_pred = self.model.predict_on_batch ( [ self.clinical[i] , self.x_validate_a[i] , self.x_validate_s[i] , self.x_validate_c[i] ]  )
            # group by patient - to get one prediction per patient only
            labelsOut,logitsOut = aggregate( self.y_validate[i] , y_pred )
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
        # before appending, check if this auc is the highest in all the lsit

        if all(auc1>i for i in self.auc):
            self.model.save_weights("/home/ubuntu/output/" + RUN + "_model.h5")
            print("Saved model to disk")

        self.auc.append(auc1)
        print ("wtf2")



        self.validation_logits.append(allLogits)
        self.train_loss.append(logs.get('loss'))
        

        # overwrite every time - no problem
        # save stuff
        np.save( "/home/ubuntu/output/" + RUN + "_auc.npy", self.auc)
        np.save( "/home/ubuntu/output/" + RUN + "_validation_logits.npy", self.validation_logits)
        np.save( "/home/ubuntu/output/" + RUN + "_train_loss.npy", self.train_loss)


        # validate loss
        #
        for i in range (self.count ):
            # now do loss
            temp = self.model.test_on_batch ( [ self.clinical[i] , self.x_validate_a[i] , self.x_validate_s[i] , self.x_validate_c[i] ]  , self.y_validate[i] )
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


def createGenerator( clincial , A, S, C, Y, batch_size, generator):

    while True:
        # suffled indices    
        idx = np.random.permutation( A.shape[0])
        # create image generator

        batches_clinical = dummyGenerator.flow( clincial[idx], Y[idx], batch_size=batch_size , shuffle=False, seed = 1)

        batches_A = generator.flow( A[idx], Y[idx], batch_size=batch_size , shuffle=False, seed = 1)
        batches_S = generator.flow( S[idx], Y[idx], batch_size=batch_size , shuffle=False, seed = 1)
        batches_C = generator.flow( C[idx], Y[idx], batch_size=batch_size , shuffle=False, seed = 1)

        print (batches_A)

        for batch_clinical , batch_a , batch_s , batch_c , counter in zip( batches_clinical , batches_A , batches_S , batches_C , np.arange(batch_size) ):

            yield [ batch_clinical[0] , batch_a[0] , batch_s[0] , batch_c[0] ] , batch_a[1]

            if counter >= batch_size:
                break





