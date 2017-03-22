import funcs
import krs
from keras.models import model_from_json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K



#
RUN = "26"
print (" testing : run: A " , RUN)
mode = "3d"
finalSize = 130
imgSize = 100
fork = True
count = 2 # only if mode 3d and fork=True
funcs.valTestMultiplier = 1
# 
skip = 3 # if fork false: imgSize/skip should be int



# get only death by disease
# all patients with cause of death 1
# diseaseDeath = np.load("rt_diseaseDeath.npy").astype(str)
# print (len(diseaseDeath))

# dataFrameTest0 = dataFrameTest[ dataFrameTest['deadstat'] == 0.0 ]
# print (dataFrameTest0.shape)

# dataFrameTest1 = dataFrameTest[ dataFrameTest['deadstat'] == 1.0 ]
# print (dataFrameTest1.shape)

# dataFrameTest1_1 = dataFrameTest1 [ dataFrameTest1["patient"].isin(diseaseDeath) ]
# print (dataFrameTest1_1.shape)

# dataFrameTest = pd.concat([dataFrameTest0,dataFrameTest1_1],axis=0)
# dataFrameTest = dataFrameTest.reset_index(drop=True)
# print dataFrameTest.shape

#
#
#
#


#
#
#       .g8"""bgd `7MM"""YMM MMP""MM""YMM     `7MM"""Yb.      db   MMP""MM""YMM   db
#     .dP'     `M   MM    `7 P'   MM   `7       MM    `Yb.   ;MM:  P'   MM   `7  ;MM:
#     dM'       `   MM   d        MM            MM     `Mb  ,V^MM.      MM      ,V^MM.
#     MM            MMmmMM        MM            MM      MM ,M  `MM      MM     ,M  `MM
#     MM.    `7MMF' MM   Y  ,     MM            MM     ,MP AbmmmqMA     MM     AbmmmqMA
#     `Mb.     MM   MM     ,M     MM            MM    ,dP'A'     VML    MM    A'     VML
#       `"bmmmdPY .JMMmmmmMMM   .JMML.        .JMMmmmdP'.AMA.   .AMMA..JMML..AMA.   .AMMA.
#
#

dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames()

#
#
#     MMP""MM""YMM `7MM"""Mq.        db      `7MMF'`7MN.   `7MF'
#     P'   MM   `7   MM   `MM.      ;MM:       MM    MMN.    M
#          MM        MM   ,M9      ,V^MM.      MM    M YMb   M
#          MM        MMmmdM9      ,M  `MM      MM    M  `MN. M
#          MM        MM  YM.      AbmmmqMA     MM    M   `MM.M
#          MM        MM   `Mb.   A'     VML    MM    M     YMM
#        .JMML.    .JMML. .JMM..AMA.   .AMMA..JMML..JML.    YM
#
#

x_train,y_train,zeros,ones,clinical_train =  funcs.getXandY(dataFrameTrain,imgSize, False)
mean,std,x_train_cs = funcs.centerAndStandardizeTraining(x_train)
print ( "mean and std shape: " ,mean.shape,std.shape )

print ("zeros: " , zeros , "ones: " , ones)
zeroWeight = ones / ((ones+zeros)*1.0)
oneWeight = zeros / ((ones+zeros)*1.0)
print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)


#
#
#     MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM
#     P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7
#          MM        MM   d    `MMb.          MM
#          MM        MMmmMM      `YMMNq.      MM
#          MM        MM   Y  , .     `MM      MM
#          MM        MM     ,M Mb     dM      MM
#        .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.
#


x_test,y_test,zeros,ones,clinical_test =  funcs.getXandY(dataFrameTest,imgSize, True)
print ("test data:" , x_test.shape,  y_test.shape  ) 

# center and standardize
x_test_cs = funcs.centerAndStandardizeValTest(x_test,mean,std)


if fork:
    # lets get the 3 orientations
    x_test_a,x_test_s,x_test_c = krs.splitValTest(x_test_cs,finalSize,imgSize,count,mode)
    print ("final val data:" , x_test_a.shape,x_test_s.shape,x_test_c.shape)

else:
    x_test = krs.splitValTest_single3D(x_test_cs,finalSize,imgSize,skip)
    print ("final val data:" , x_test.shape)


#
#
#       .g8"""bgd `7MM"""YMM MMP""MM""YMM     `7MMM.     ,MMF' .g8""8q. `7MM"""Yb. `7MM"""YMM  `7MMF'
#     .dP'     `M   MM    `7 P'   MM   `7       MMMb    dPMM .dP'    `YM. MM    `Yb. MM    `7    MM
#     dM'       `   MM   d        MM            M YM   ,M MM dM'      `MM MM     `Mb MM   d      MM
#     MM            MMmmMM        MM            M  Mb  M' MM MM        MM MM      MM MMmmMM      MM
#     MM.    `7MMF' MM   Y  ,     MM            M  YM.P'  MM MM.      ,MP MM     ,MP MM   Y  ,   MM      ,
#     `Mb.     MM   MM     ,M     MM            M  `YM'   MM `Mb.    ,dP' MM    ,dP' MM     ,M   MM     ,M
#       `"bmmmdPY .JMMmmmmMMM   .JMML.        .JML. `'  .JMML. `"bmmd"' .JMMmmmdP' .JMMmmmmMMM .JMMmmmmMMM
#
#

# load json and create model
json_file = open( "/home/ubuntu/output/" + RUN + '_json.json' , 'r')
loaded_model_json = json_file.read()
json_file.close()
myModel = model_from_json(loaded_model_json)
# load weights into new model
myModel.load_weights("/home/ubuntu/output/" + RUN + "_model.h5")



#
#
#     `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF' .g8"""bgd  .M"""bgd
#       MM    `7   MM       M    MMN.    M .dP'     `M ,MI    "Y
#       MM   d     MM       M    M YMb   M dM'       ` `MMb.
#       MM""MM     MM       M    M  `MN. M MM            `YMMNq.
#       MM   Y     MM       M    M   `MM.M MM.         .     `MM
#       MM         YM.     ,M    M     YMM `Mb.     ,' Mb     dM
#     .JMML.        `bmmmmd"'  .JML.    YM   `"bmmmd'  P"Ybmmd"
#
#

if fork:

    # (0 = test, 1 = train) 
    axialFunc = K.function([ myModel.layers[0].layers[0].layers[0].input , K.learning_phase()  ], 
                       [ myModel.layers[0].layers[0].layers[-1].output ] )

    sagittalFunc = K.function([ myModel.layers[0].layers[1].layers[0].input , K.learning_phase()  ], 
                       [ myModel.layers[0].layers[1].layers[-1].output ] )

    coronalFunc = K.function([ myModel.layers[0].layers[2].layers[0].input , K.learning_phase()  ], 
                       [ myModel.layers[0].layers[2].layers[-1].output ] )

    mergeFunc = K.function([ myModel.layers[1].input , K.learning_phase()  ], 
                       [ myModel.layers[2].output ] ) 

    softmaxFunc = K.function([ myModel.layers[3].input , K.learning_phase()  ], 
                       [ myModel.layers[3].output ] )


else:

    print("no fork - not tested")


#
#
#     `7MMF'        .g8""8q.     .g8""8q. `7MM"""Mq.
#       MM        .dP'    `YM. .dP'    `YM. MM   `MM.
#       MM        dM'      `MM dM'      `MM MM   ,M9
#       MM        MM        MM MM        MM MMmmdM9
#       MM      , MM.      ,MP MM.      ,MP MM
#       MM     ,M `Mb.    ,dP' `Mb.    ,dP' MM
#     .JMMmmmmMMM   `"bmmd"'     `"bmmd"' .JMML.
#
#

logits = []
#
for i in range (dataFrameTest.shape[0]):


    if fork:

        if mode == "2d":
            # get the different ones
            axial512 = axialFunc( [  x_test_a[i].reshape(1,imgSize,imgSize,1) , 0 ] )
            sagittal512 = sagittalFunc( [  x_test_s[i].reshape(1,imgSize,imgSize,1) , 0 ] )
            coronal512 = coronalFunc( [  x_test_c[i].reshape(1,imgSize,imgSize,1) , 0 ] )
        if mode == "3d":
            axial512 = axialFunc( [  x_test_a[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ] )
            sagittal512 = sagittalFunc( [  x_test_s[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ] )
            coronal512 = coronalFunc( [  x_test_c[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ] )

        # concat them
        concat = []
        concat.extend ( axial512[0][0].tolist() )
        concat.extend ( sagittal512[0][0].tolist() )
        concat.extend ( coronal512[0][0].tolist() )
        #
        concat = np.array(concat ,'float32').reshape(1,len(concat))
        # now do one last function
        preds = mergeFunc( [ concat , 0 ])
        print (   preds[0][0][0] ,  preds[0][0][1]   )
        #
        logitsBal = np.array( [ preds[0][0][0] * zeroWeight ,  preds[0][0][1] * oneWeight ]  ) .reshape(1,2)
        logits.append(  softmaxFunc(   [ logitsBal     , 0 ]) [0].reshape(2)  )

    else:

        print("no fork - not tested")


logits = np.array(logits)
print ("logits: " , logits.shape , logits[0] , logits[30] , logits[60]  )
auc1 , auc2 = funcs.AUC(  y_test ,  logits )
print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
print ("wtf2")



##############################################################################################################################################



#     if fork:
#         # get predictions
#         y_pred = myModel.predict_on_batch ( [ clinical_test[i] , x_test_a[i] , x_test_s[i] , x_test_c[i] ]  )
#     else:
#         y_pred = myModel.predict_on_batch ( [ clinical_test[i] , x_test[i] ]  )

#     # save raw logits
#     rawLogits.extend( y_pred  ) 
#     # group by patient - to get one prediction per patient only
#     labelsOut,logitsOut = funcs.aggregate( y_test[i] , y_pred )
#     #
#     allLabels.extend(labelsOut)
#     allLogits.extend(logitsOut)
#     #


# # save logits
# np.save( "/home/ubuntu/output/" + RUN + "_test_logits_raw.npy", rawLogits )


# allLabels = np.array(allLabels)
# allLogits = np.array(allLogits)

# print ("\nfinal labels,logits shape: " , allLabels.shape , allLogits.shape ) # 528


# # get 2 auc's
# print ("wtf1")
# auc1 , auc2 = funcs.AUC(  allLabels ,  allLogits )
# print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
# # before appending, check if this auc is the highest in all the lsit
# print ("wtf2")




