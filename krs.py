
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
#           db   `7MMF'   `7MF' .g8"""bgd `7MMM.     ,MMF'`7MM"""YMM  `7MN.   `7MF'MMP""MM""YMM
#          ;MM:    MM       M .dP'     `M   MMMb    dPMM    MM    `7    MMN.    M  P'   MM   `7
#         ,V^MM.   MM       M dM'       `   M YM   ,M MM    MM   d      M YMb   M       MM
#        ,M  `MM   MM       M MM            M  Mb  M' MM    MMmmMM      M  `MN. M       MM
#        AbmmmqMA  MM       M MM.    `7MMF' M  YM.P'  MM    MM   Y  ,   M   `MM.M       MM
#       A'     VML YM.     ,M `Mb.     MM   M  `YM'   MM    MM     ,M   M     YMM       MM
#     .AMA.   .AMMA.`bmmmmd"'   `"bmmmdPY .JML. `'  .JMML..JMMmmmmMMM .JML.    YM     .JMML.
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

# these 4 functions below apply rotations during augmentation
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_rotation(x, theta=90, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def applyRotation(arr,theta):
    #
    out = np.copy(arr)
    #
    for i in range (arr.shape[0]):
        out[i] = random_rotation(out[i],theta = theta, 
                                 row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect', cval=0. )
    return  out




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
            # angleList = [-180,-90,0,90,180]
            # randAng = angleList [ random.randint(0,4) ]
            # theta = np.pi / 180 *  randAng # 
            # # option 1 :  np.random.uniform(-180, 180) 
            # # option 2 : randAng 
            # miniPatch = applyRotation(miniPatch,theta) 

            # alternate faster numpy rotation
            # uses numpy 1.12.1
            # posotive is counter-clockwise , negative is clockwise .. but same thing..
            # 0 = no rotation
            # list to choose from [0,1,2,3]

            #
            #
            #

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
                # arr_list.append (  miniPatch [  0:imgSize:skip , 0:imgSize:skip  , 0:imgSize:skip  ]  )
                arr_list.append (  miniPatch [ (mid-travel) : (mid+travel+1) : skip ,:,:]  )

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
                # arr_list.append (  miniPatch [  0:imgSize:skip , 0:imgSize:skip  , 0:imgSize:skip  ] )
                arr_list.append (  miniPatch [ (mid-travel) : (mid+travel+1) : skip ,:,:]  )
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
                    # arr_list.append (  miniPatch [  0:imgSize:skip , 0:imgSize:skip  , 0:imgSize:skip  ] )
                    arr_list.append (  miniPatch [ (mid-travel) : (mid+travel+1) : skip ,:,:]  )
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