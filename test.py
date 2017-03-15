import funcs
import krs
from keras.models import model_from_json
import numpy as np
import pandas as pd


#
RUN = "17"
print (" testing : run: A " , RUN)
mode = "2d"
finalSize = 150
imgSize = 120
fork = True
count = 200 # only if mode 3d and fork=True
funcs.valTestMultiplier = 1
#
chunkCount = 3
chunkReduced = 45
# 
skip = 3 # if fork false: imgSize/skip should be int

# get dataframes
dataFrameTrain,dataFrameValidate,dataFrameTest= funcs.manageDataFrames()

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


# get mean and std from training first
x_train,y_train,zeros,ones,clinical_train =  funcs.getXandY(dataFrameTrain,imgSize, False)
mean,std,x_train_cs = funcs.centerAndStandardizeTraining(x_train)
print ( "mean and std shape: " ,mean.shape,std.shape )

# get test data
x_test,y_test,zeros,ones,clinical_test =  funcs.getXandY(dataFrameTest,imgSize, False)
print ("train data:" , x_test.shape,  y_test.shape , clinical_test.shape ) 


# center and standardize
x_test_cs = funcs.centerAndStandardizeValTest(x_test,mean,std)


if fork:
    # lets get the 3 orientations
    x_test_a,x_test_s,x_test_c = krs.splitValTest(x_test_cs,finalSize,imgSize,count,mode)
    print ("final val data:" , x_test_a.shape,x_test_s.shape,x_test_c.shape)

    # now lets break them into chuncks 
    y_test = funcs.getChuncks(y_test, chunkCount , chunkReduced)
    x_test_a = funcs.getChuncks(x_test_a, chunkCount, chunkReduced)   
    x_test_s = funcs.getChuncks(x_test_s, chunkCount , chunkReduced)
    x_test_c = funcs.getChuncks(x_test_c, chunkCount , chunkReduced)
    clinical_test = funcs.getChuncks(clinical_test, chunkCount , chunkReduced)

    print ("part of validate data: " , x_test_a[0].shape )
    print ("part of validate labels: " , y_test[0].shape )

else:
    x_test = krs.splitValTest_single3D(x_test_cs,finalSize,imgSize,skip)

    print ("final val data:" , x_test.shape)

    # now lets break them into chuncks divisible by 9 to fit into the GPU
    y_test = getChuncks(y_test, chunkCount , chunkReduced)
    x_test = getChuncks(x_test, chunkCount , chunkReduced)   
    clinical_test = getChuncks(clinical_test, chunkCount , chunkReduced)

    print ("part of validate data: " , x_test[0].shape )
    print ("part of validate labels: " , y_test[0].shape )

# get the model
# load json and create model
json_file = open( "/home/ubuntu/output/" + RUN + '_json.json' , 'r')
loaded_model_json = json_file.read()
json_file.close()
myModel = model_from_json(loaded_model_json)
# load weights into new model
myModel.load_weights("/home/ubuntu/output/" + RUN + "_model.h5")


# get AUC

allLabels = []
allLogits = []
rawLogits = []
#
for i in range (chunkCount ):

    if fork:
        # get predictions
        y_pred = myModel.predict_on_batch ( [ clinical_test[i] , x_test_a[i] , x_test_s[i] , x_test_c[i] ]  )
    else:
        y_pred = myModel.predict_on_batch ( [ clinical_test[i] , x_test[i] ]  )

    # save raw logits
    rawLogits.extend( y_pred  ) 
    # group by patient - to get one prediction per patient only
    labelsOut,logitsOut = funcs.aggregate( y_test[i] , y_pred )
    #
    allLabels.extend(labelsOut)
    allLogits.extend(logitsOut)
    #


# save logits
np.save( "/home/ubuntu/output/" + RUN + "_test_logits_raw.npy", rawLogits )


allLabels = np.array(allLabels)
allLogits = np.array(allLogits)

print ("\nfinal labels,logits shape: " , allLabels.shape , allLogits.shape ) # 528


# get 2 auc's
print ("wtf1")
auc1 , auc2 = funcs.AUC(  allLabels ,  allLogits )
print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)
# before appending, check if this auc is the highest in all the lsit
print ("wtf2")




