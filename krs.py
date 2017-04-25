
from __future__ import division

import numpy as np
np.random.seed(123)
import scipy.ndimage as ndi
import random 
from keras.utils import np_utils
import scipy.misc
import time


#
#
#       .g8"""bgd `7MM"""YMM  `7MN.   `7MF'`7MM"""YMM  `7MM"""Mq.        db   MMP""MM""YMM   .g8""8q. `7MM"""Mq.
#     .dP'     `M   MM    `7    MMN.    M    MM    `7    MM   `MM.      ;MM:  P'   MM   `7 .dP'    `YM. MM   `MM.
#     dM'       `   MM   d      M YMb   M    MM   d      MM   ,M9      ,V^MM.      MM      dM'      `MM MM   ,M9
#     MM            MMmmMM      M  `MN. M    MMmmMM      MMmmdM9      ,M  `MM      MM      MM        MM MMmmdM9
#     MM.    `7MMF' MM   Y  ,   M   `MM.M    MM   Y  ,   MM  YM.      AbmmmqMA     MM      MM.      ,MP MM  YM.
#     `Mb.     MM   MM     ,M   M     YMM    MM     ,M   MM   `Mb.   A'     VML    MM      `Mb.    ,dP' MM   `Mb.
#       `"bmmmdPY .JMMmmmmMMM .JML.    YM  .JMMmmmmMMM .JMML. .JMM..AMA.   .AMMA..JMML.      `"bmmd"' .JMML. .JMM.
#
#

# augments, randmoizes and splits into batches.
# 
def augmentAndSplitTrain(x_train,y_train,finalSize,imgSize,count, batchSize, mode, fork, skip): 
    
    # for forking
    arr_a_list = []
    arr_s_list = []
    arr_c_list = []
    # for no forking
    arr_list = []

    counter = 0
    # loop through each patient.
    for arr in iter(x_train):
        #

        
        # offset array to get a smaller one
        offsetArr = offsetPatch(arr, finalSize)

        # get random miniPatch 
        offstX = random.randint(0,finalSize-imgSize)
        offstY = random.randint(0,finalSize-imgSize)
        offstZ = random.randint(0,finalSize-imgSize)
        miniPatch = offsetArr[offstX:imgSize+offstX,offstY:imgSize+offstY,offstZ:imgSize+offstZ]

        # rand = random.randint(0,8)
        # miniPatch = getMiniPatch( rand , offsetArr , imgSize )

        # reshape to make one channel
        miniPatch = miniPatch.reshape(imgSize,imgSize,imgSize,1)



        # if we want to augment the data
        if augmentTraining:

            # flip bools
            flipBoolud = bool(random.getrandbits(1))
            flipBoolio = bool(random.getrandbits(1))
            flipBoolrl = bool(random.getrandbits(1))

            #
            if flipBoolud:
                miniPatch =  np.fliplr(miniPatch)    
            if flipBoolio:
                miniPatch =  np.flipud(miniPatch) 
            if flipBoolrl:
                miniPatch =  miniPatch[:,:,::-1]

            # # rotation
            miniPatch = np.rot90(miniPatch , k= random.randint(0,3) , axes=(1,2) )

            # OTHER AUGMENTATIONS COME HERE .....


        # EXTRACT ORIENTATION SLICES
        travel = int(count * skip)
        mid  = int(imgSize/2.0)

        # if we are forking - either 2d or 3d
        if fork:

            if mode == "3d":
                #
                arr_a_list.append( miniPatch [(mid-travel):(mid+travel+1):skip,:,:] )
                #
                arr_s = miniPatch [:,:,(mid-travel):(mid+travel+1):skip]
                arr_s_list.append( np.swapaxes( np.rot90(arr_s,3) , 0,2).reshape(count*2+1,imgSize,imgSize,1)  )
                #
                arr_c = miniPatch [:,(mid-travel):(mid+travel+1):skip,:]
                arr_c_list.append( np.swapaxes(np.flipud (arr_c) ,0,1).reshape(count*2+1,imgSize,imgSize,1) )

            elif mode == "2d":
                #
                arr_a_list.append( miniPatch [mid,:,:] )
                arr_s_list.append( np.flipud ( miniPatch [:,:,mid] ) )
                arr_c_list.append( np.flipud ( miniPatch [:,mid,:] ) )

                # for saving images - ONLY 2d
                # scipy.misc.imsave("/home/ahmed/output/imgtst/" + str(counter) + "_A.png" , miniPatch [mid,:,:].reshape(imgSize,imgSize) )
                # scipy.misc.imsave("/home/ahmed/output/imgtst/" + str(counter+1) + "_S.png" , np.flipud ( miniPatch [:,:,mid] ).reshape(imgSize,imgSize) )
                # scipy.misc.imsave("/home/ahmed/output/imgtst/" + str(counter+2) + "_C.png" , np.flipud ( miniPatch [:,mid,:] ).reshape(imgSize,imgSize) )
                # counter = counter+3

        # if single i.e. no forking
        else: 

            if mode == "3d":
                # append as is - skipping already done before augmentation to speed things up
                arr_list.append (  miniPatch [  0:imgSize:skip , 0:imgSize:skip  , 0:imgSize:skip  ]  )
                # arr_list.append (  miniPatch [ (mid-travel) : (mid+travel+1) : skip ,:,:]  )

            elif mode == "2d":
                arr_list.append (  miniPatch [mid,:,:] ) # only axial


    #
    # AFTER LOOP 
    #
    if fork:        
        # RANDOMIZE
        idx = np.random.permutation( len(x_train))
        # all is 5, batch size is 2, batch proper is 4, no of batches is 2
        batchProper = len(x_train) - (len(x_train)%batchSize) 
        noOfBatches = batchProper / batchSize
        # reorder all and take first batch*int entries i.e. leave remainder out, then split
        a_train = np.split  (   np.array( arr_a_list, 'float32') [idx] [:batchProper]   , noOfBatches )
        s_train = np.split  (   np.array( arr_s_list, 'float32') [idx] [:batchProper]   , noOfBatches )
        c_train = np.split  (   np.array( arr_c_list, 'float32') [idx] [:batchProper]   , noOfBatches )
        # now the label
        y_train_out = np.split  (     y_train                    [idx] [:batchProper]   , noOfBatches )     

        return a_train,s_train,c_train,y_train_out 

    else:

        # AFTER LOOP
        # RANDOMIZE
        idx = np.random.permutation( len(x_train))
        # all is 5, batch size is 2, batch proper is 4, no of batches is 2
        batchProper = len(x_train) - (len(x_train)%batchSize) 
        noOfBatches = batchProper / batchSize
        # reorder all and take first batch*int entries i.e. leave remainder out, then split
        x_train_new = np.split  (   np.array( arr_list, 'float32') [idx] [:batchProper]   , noOfBatches )
        #
        y_train_out = np.split  (     y_train                     [idx] [:batchProper]   , noOfBatches )     

        return x_train_new,y_train_out




# runs every epoch
def myGenerator(x_train,y_train,finalSize,imgSize,count,batchSize,mode,fork,skip): # clinical_train,

    while True:
        
        #####
        if fork:
            # these are acually lists of batches
            a_train,s_train,c_train,y_train_out = augmentAndSplitTrain(x_train,y_train,finalSize,imgSize,count,batchSize,mode,fork,skip)

            batches = 0
            for   _a_train,_s_train,_c_train,_y_train in zip(
                a_train,s_train,c_train,y_train_out): 

                yield [ _a_train ,_s_train , _c_train ] , _y_train  

                batches += 1
                if batches ==  len(a_train) :
                    break

        #### 
        else:
            # these are acually lists of batches
            x_train_out,y_train_out = augmentAndSplitTrain(x_train,y_train,finalSize,imgSize,count,batchSize,mode,fork,skip)

            batches = 0
            for   _x_train,_y_train in zip( 
                x_train_out,y_train_out ):

                yield [ _x_train ] , _y_train   

                batches += 1
                if batches ==  len(x_train_out) :
                    break
        



# this is used to extract the slices (either 2d or 3d. fork or no fork) from the validation or test sets
# for training, this is automatically done in the generator
def splitValTest(x_valTest,finalSize,imgSize,count,mode,fork,skip):

    # for forking
    arr_a_list = []
    arr_s_list = []
    arr_c_list = []
    # for no forking
    arr_list = []

    # loop through each patient.
    for arr in iter(x_valTest):

        # offset array to get a smaller one
        offsetArr = offsetPatch(arr, finalSize)

        # gets the patch at the center
        miniPatch = getMiniPatch(4,offsetArr,imgSize)

        # reshape to make channel
        miniPatch = miniPatch.reshape(imgSize,imgSize,imgSize,1)


        # EXTRACT ORIENTATION SLICES
        travel = int(count * skip)
        mid  = int(imgSize/2.0)

        if fork:

            if mode == "3d":
                #
                arr_a_list.append( miniPatch [(mid-travel):(mid+travel+1):skip,:,:] )
                #
                arr_s = miniPatch [:,:,(mid-travel):(mid+travel+1):skip]
                arr_s_list.append( np.swapaxes( np.rot90(arr_s,3) , 0,2).reshape(count*2+1,imgSize,imgSize,1)  )
                #
                arr_c = miniPatch [:,(mid-travel):(mid+travel+1):skip,:]
                arr_c_list.append( np.swapaxes(np.flipud (arr_c) ,0,1).reshape(count*2+1,imgSize,imgSize,1) )

            elif mode == "2d":
                #
                arr_a_list.append( miniPatch [mid,:,:] )
                arr_s_list.append( np.flipud ( miniPatch [:,:,mid] ) )
                arr_c_list.append( np.flipud ( miniPatch [:,mid,:] ) )

        else:

            if mode == "3d":
                # EXTRACT SINGLE
                arr_list.append (  miniPatch [  0:imgSize:skip , 0:imgSize:skip  , 0:imgSize:skip  ] )
                # arr_list.append (  miniPatch [ (mid-travel) : (mid+travel+1) : skip ,:,:]  )
            elif mode == "2d":
                arr_list.append (  miniPatch [mid,:,:] ) # only axial


    #
    # AFTER LOOP - no randomizing here - just as is
    #
    if fork:        
        # 
        arr_a_list = np.array( arr_a_list ) 
        arr_s_list = np.array( arr_s_list )
        arr_c_list = np.array( arr_c_list )  
               
        return arr_a_list,arr_s_list,arr_c_list

    else:
        #
        arr_list = np.array( arr_list ) 
      
        return arr_list


# this is used to extract the slices (either 2d or 3d. fork or no fork) from the validation or test sets
# for training, this is automatically done in the generator
# this one exports multiple
def splitValTestMul(x_valTest,finalSize,imgSize,count,mode,fork,skip):

    # for forking
    arr_a_list = []
    arr_s_list = []
    arr_c_list = []
    # for no forking
    arr_list = []

    # loop through each patient.
    for arr in iter(x_valTest):

        # offset array to get a smaller one
        offsetArr = offsetPatch(arr, finalSize)


        for j in range(0,9):

            # gets the patch at the center
            miniPatch = getMiniPatch(j,offsetArr,imgSize)

            # reshape to make channel
            miniPatch = miniPatch.reshape(imgSize,imgSize,imgSize,1)


            # EXTRACT ORIENTATION SLICES
            travel = int(count * skip)
            mid  = int(imgSize/2.0)

            if fork:

                if mode == "3d":
                    #
                    arr_a_list.append( miniPatch [(mid-travel):(mid+travel+1):skip,:,:] )
                    #
                    arr_s = miniPatch [:,:,(mid-travel):(mid+travel+1):skip]
                    arr_s_list.append( np.swapaxes( np.rot90(arr_s,3) , 0,2).reshape(count*2+1,imgSize,imgSize,1)  )
                    #
                    arr_c = miniPatch [:,(mid-travel):(mid+travel+1):skip,:]
                    arr_c_list.append( np.swapaxes(np.flipud (arr_c) ,0,1).reshape(count*2+1,imgSize,imgSize,1) )

                elif mode == "2d":
                    #
                    arr_a_list.append( miniPatch [mid,:,:] )
                    arr_s_list.append( np.flipud ( miniPatch [:,:,mid] ) )
                    arr_c_list.append( np.flipud ( miniPatch [:,mid,:] ) )

            else:

                if mode == "3d":
                    # EXTRACT SINGLE
                    arr_list.append (  miniPatch [  0:imgSize:skip , 0:imgSize:skip  , 0:imgSize:skip  ] )
                    # arr_list.append (  miniPatch [ (mid-travel) : (mid+travel+1) : skip ,:,:]  )
                elif mode == "2d":
                    arr_list.append (  miniPatch [mid,:,:] ) # only axial



    #
    # AFTER LOOP - no randomizing here - just as is
    #
    if fork:        
        # 
        arr_a_list = np.array( arr_a_list ) 
        arr_s_list = np.array( arr_s_list )
        arr_c_list = np.array( arr_c_list )  
               
        return arr_a_list,arr_s_list,arr_c_list

    else:
        #
        arr_list = np.array( arr_list ) 
      
        return arr_list




#
#
#     `7MMF'  `7MMF'`7MM"""YMM  `7MMF'      `7MM"""Mq.`7MM"""YMM  `7MM"""Mq.   .M"""bgd
#       MM      MM    MM    `7    MM          MM   `MM. MM    `7    MM   `MM. ,MI    "Y
#       MM      MM    MM   d      MM          MM   ,M9  MM   d      MM   ,M9  `MMb.
#       MMmmmmmmMM    MMmmMM      MM          MMmmdM9   MMmmMM      MMmmdM9     `YMMNq.
#       MM      MM    MM   Y  ,   MM      ,   MM        MM   Y  ,   MM  YM.   .     `MM
#       MM      MM    MM     ,M   MM     ,M   MM        MM     ,M   MM   `Mb. Mb     dM
#     .JMML.  .JMML..JMMmmmmMMM .JMMmmmmMMM .JMML.    .JMMmmmmMMM .JMML. .JMM.P"Ybmmd"
#
#

# this offsets the 150x150x150 patch into a smaller one equally from all sides
def offsetPatch(arr, finalSize):
    offset = int( (150-finalSize) / 2.0 )
    offsetEnd = int ( 150-offset )
    return arr[ offset:offsetEnd , offset:offsetEnd , offset:offsetEnd ]
    
# this function is used only during val/test
# it gets the middle patch by feeding "4" into rand
def getMiniPatch(rand,arr,imgSize):
    lower = arr.shape[0] - imgSize
    maxi = arr.shape[0]
    mid1 = int(lower/2.0)
    mid2 = maxi-int(lower/2.0)
    
    if rand == 0:
        return arr[0:imgSize,0:imgSize,0:imgSize]
    elif rand == 1:
        return arr[0:imgSize,0:imgSize,lower:maxi]
    elif rand == 2:
        return arr[0:imgSize,lower:maxi,lower:maxi]  
    elif rand == 3:
        return arr[0:imgSize,lower:maxi,0:imgSize]  
    #
    elif rand == 4:
        return arr[mid1:mid2,mid1:mid2,mid1:mid2] 
    #
    elif rand == 5:
        return arr[lower:maxi,0:imgSize,0:imgSize]
    elif rand == 6:
        return arr[lower:maxi,0:imgSize,lower:maxi]
    elif rand == 7:
        return arr[lower:maxi,lower:maxi,lower:maxi]  
    elif rand == 8:
        return arr[lower:maxi,lower:maxi,0:imgSize]  

