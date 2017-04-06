import funcs
import krs
from keras.models import model_from_json
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K


# current version
RUN = "55"
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

# print 
print ("training : run: " , RUN )




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

# x_train,y_train,zeros,ones =  funcs.getXandY(dataFrameTrain,imgSize)
# mean,std,x_train_cs = funcs.centerAndStandardizeTraining(x_train)
# print ( "mean and std shape: " ,mean.shape,std.shape )

# print ("zeros: " , zeros , "ones: " , ones)
# zeroWeight = ones / ((ones+zeros)*1.0)
# oneWeight = zeros / ((ones+zeros)*1.0)
# print ("zeroWeight: " , zeroWeight , "oneWeight: " , oneWeight)



#
#     MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM
#     P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7
#          MM        MM   d    `MMb.          MM
#          MM        MMmmMM      `YMMNq.      MM
#          MM        MM   Y  , .     `MM      MM
#          MM        MM     ,M Mb     dM      MM
#        .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.
#

# get dataframe first

testList = ["lung2"] # split to val and test

dataFrameTest = pd.DataFrame.from_csv('master_170228.csv', index_col = 0)
dataFrameTest = dataFrameTest [ 
    ( pd.notnull( dataFrameTest["pathToData"] ) ) &
    ( pd.notnull( dataFrameTest["pathToMask"] ) ) &
    ( pd.notnull( dataFrameTest["stackMin"] ) ) &
    ( pd.isnull( dataFrameTest["patch_failed"] ) ) 
    ]

dataFrameTest = dataFrameTest [ dataFrameTest["dataset"].isin(testList) ]
dataFrameTest = dataFrameTest.reset_index(drop=True)

x_test =  funcs.getX(dataFrameTest,imgSize)
print ("test data:" , x_test.shape ) 

# center and standardize
# x_test_cs = funcs.centerAndStandardizeValTest(x_test,mean,std)
x_test_cs = funcs.centerAndNormalize(x_test)

if fork:
    # lets get the 3 orientations
    x_test_a,x_test_s,x_test_c = krs.splitValTest(x_test_cs,finalSize,imgSize,count,mode,fork,skip)
    print ("final val data:" , x_test_a.shape,x_test_s.shape,x_test_c.shape)

else:
    x_test = krs.splitValTest(x_test_cs,finalSize,imgSize,count,mode,fork,skip)
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


# make funcs
# to befire last dense - size 256
func1 = K.function([ myModel.layers[0].input , K.learning_phase()  ], [ myModel.layers[21].output ] )
# to last dense - size 2
func2 = K.function([ myModel.layers[0].input , K.learning_phase()  ], [ myModel.layers[25].output ] )


func1List = []
func2List = []

for i in range (dataFrameTest.shape[0]):

    if fork: 

        if mode == "3d":
            inputa = [ x_test_a[i].reshape(1,count*2+1,imgSize,imgSize,1) , 
                x_test_s[i].reshape(1,count*2+1,imgSize,imgSize,1) , 
                x_test_c[i].reshape(1,count*2+1,imgSize,imgSize,1) , 0 ]

            func1out = func1( inputa )
            func2out = func2( inputa )

        elif mode == "2d":
            inputa = [ x_test_a[i].reshape(1,imgSize,imgSize,1) , 
                x_test_s[i].reshape(1,imgSize,imgSize,1) , 
                x_test_c[i].reshape(1,imgSize,imgSize,1) , 0 ]

            func1out = func1( inputa )
            func2out = func2( inputa )

    else:

        if mode == "3d":
            inputa =  [ x_test[i].reshape(1,imgSize/skip,imgSize/skip,imgSize/skip,1) , 0 ]
            func1out = func1( inputa )
            func2out = func2( inputa )

        elif mode == "2d":
            inputa =  [ x_test[i].reshape(1,imgSize,imgSize,1) , 0 ]
            func1out = func1( inputa )
            func2out = func2( inputa )


    func1List.append(func1out[0][0])
    func2List.append(func2out[0][0])

    
func1List = np.array(func1List).transpose()
func2List = np.array(func2List).transpose()

print ( "funcs transpose shape: " , func1List.shape , func2List.shape )

for i in range ( func1List.shape[0] ):
    dataFrameTest['dense1_' + str (i)] = func1List[i]

for i in range ( func2List.shape[0] ):
    dataFrameTest['dense2_' + str (i)] = func2List[i]


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

        if mode == "3d":
            # get predictions
            y_pred = myModel.predict_on_batch ( [ x_test_a[i].reshape(1,count*2+1,imgSize,imgSize,1) , 
                x_test_s[i].reshape(1,count*2+1,imgSize,imgSize,1) , 
                x_test_c[i].reshape(1,count*2+1,imgSize,imgSize,1) ]  )

        elif mode == "2d":
            # get predictions
            y_pred = myModel.predict_on_batch ( [ x_test_a[i].reshape(1,imgSize,imgSize,1) ,
                x_test_s[i].reshape(1,imgSize,imgSize,1) , 
                x_test_c[i].reshape(1,imgSize,imgSize,1) ]  )

    else:

        if mode == "3d":
            # get predictions
            y_pred = myModel.predict_on_batch ( [ x_test[i].reshape(1,imgSize/skip,imgSize/skip,imgSize/skip,1) ] ) 

        elif mode == "2d":
            # get predictions
            y_pred = myModel.predict_on_batch ( [ x_test[i].reshape(1,imgSize,imgSize,1) ] )


    print ( y_pred [0] )
    # now after down with switching
    logits.append( y_pred[0] )


# add logit column
dataFrameTest['logit_0'] = [ x[0] for x in logits ]
dataFrameTest['logit_1'] = [ x[1] for x in logits ]



dataFrameTest.to_csv("/home/ubuntu/output/" + RUN + "_dataFrame.csv" )





























# # after loop

# print ( "\npredicted val zeros: "  , len( [ x for x in  logits if x[0] > x[1]  ] )  )
# print ( "predicted val ones: "  , len( [ x for x in  logits if x[0] < x[1]  ] )  )


# logits = np.array(logits)
# # save logits
# np.save( "/home/ubuntu/output/" + RUN + "_test_logits.npy", logits )

# print ("logits: " , logits.shape , logits[0] , logits[30] , logits[60]  )
# auc1 , auc2 = funcs.AUC(  y_test ,  logits )
# print ("\nauc1: " , auc1 , "  auc2: " ,  auc2)





