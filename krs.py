
from __future__ import division

import numpy as np
import scipy.ndimage as ndi
import random 
from keras.utils import np_utils

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

def offsetPatch(arr, finalSize):
    offset = int( (150-finalSize) / 2.0 )
    offsetEnd = int ( 150-offset )
    return arr[ offset:offsetEnd , offset:offsetEnd , offset:offsetEnd ]
    

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
#     MMP""MM""YMM `7MM"""Mq.        db      `7MMF'`7MN.   `7MF'
#     P'   MM   `7   MM   `MM.      ;MM:       MM    MMN.    M
#          MM        MM   ,M9      ,V^MM.      MM    M YMb   M
#          MM        MMmmdM9      ,M  `MM      MM    M  `MN. M
#          MM        MM  YM.      AbmmmqMA     MM    M   `MM.M
#          MM        MM   `Mb.   A'     VML    MM    M     YMM
#        .JMML.    .JMML. .JMM..AMA.   .AMMA..JMML..JML.    YM
#
#

# augments, randmoizes and splits into batches.
def augmentAndSplitTrain(x_train,y_train,clinical_train,finalSize,imgSize,count, batchSize):
    

    arr_a_list = []
    arr_s_list = []
    arr_c_list = []

    # loop through each patient.
    for arr in iter(x_train):

        # offset array to get a smaller one
        offsetArr = offsetPatch(arr, finalSize)

        # get miniPatch (0~9)
        randInt = random.randint(0,8)
        miniPatch = getMiniPatch(randInt,offsetArr,imgSize)

        # reshape to make channel
        miniPatch = miniPatch.reshape(imgSize,imgSize,imgSize,1)

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

        # rotation
        theta = np.pi / 180 * np.random.uniform(-180, 180)
        #
        miniPatch = applyRotation(miniPatch,theta) 

        # OTHER AUGMENTATIONS COME HERE

        # EXTRACT ORIENTATION SLICES
        skip = 4 # in mm or pixel
        travel = int(count * skip)
        mid  = int(imgSize/2.0)
        #
        arr_a_list.append( miniPatch [(mid-travel):(mid+travel+1):skip,:,:] )
        #
        arr_s = miniPatch [:,:,(mid-travel):(mid+travel+1):skip]
        arr_s_list.append( np.swapaxes( np.rot90(arr_s,3) , 0,2).reshape(count*2+1,imgSize,imgSize,1)  )
        #
        arr_c = miniPatch [:,(mid-travel):(mid+travel+1):skip,:]
        arr_c_list.append( np.swapaxes(np.flipud (arr_c) ,0,1).reshape(count*2+1,imgSize,imgSize,1) )
        
    # AFTER LOOP
    # RANDOMIZE
    idx = np.random.permutation( len(x_train))
    # all is 5, batch size is 2, batch proper is 4, no of batches is 2
    batchProper = len(x_train) - (len(x_train)%batchSize) 
    noOfBatches = batchProper / batchSize
    # reorder all and take first batch*int entries i.e. leave remainder out, then split
    a_train = np.split  (   np.array( arr_a_list, 'float32') [idx] [:batchProper]   , noOfBatches )
    s_train = np.split  (   np.array( arr_s_list, 'float32') [idx] [:batchProper]   , noOfBatches )
    c_train = np.split  (   np.array( arr_c_list, 'float32') [idx] [:batchProper]   , noOfBatches )

    # print (  y_train.shape ) 

    y_train_out = np.split  (     y_train                     [idx] [:batchProper]   , noOfBatches )     
    clinical_train_out = np.split  (   clinical_train         [idx] [:batchProper]   , noOfBatches )  


    return a_train,s_train,c_train,y_train_out,clinical_train_out

# runs every epoch
def myGenerator(x_train,y_train,clinical_train,finalSize,imgSize,count,batchSize):


    while True:
        
        # these are acually lists of batches
        a_train,s_train,c_train,y_train_out,clinical_train_out = augmentAndSplitTrain(x_train,y_train,clinical_train,
                                                                 finalSize,imgSize,count,batchSize)

        # print ("final train batch data:" , a_train[0].shape,s_train[0].shape,c_train[0].shape,y_train_out[0].shape,clinical_train_out[0].shape)
        
        batches = 0
        for   _a_train,_s_train,_c_train,_y_train,_clinical_train in zip( 
            a_train,s_train,c_train,y_train_out,clinical_train_out ):

            yield [_clinical_train , _a_train , _s_train , _c_train ] , _y_train   

            batches += 1
            if batches ==  len(a_train) :
                break



#
#                                             AW
#     `7MMF'   `7MF' db      `7MMF'          ,M'MMP""MM""YMM `7MM"""YMM   .M"""bgd MMP""MM""YMM
#       `MA     ,V  ;MM:       MM            MV P'   MM   `7   MM    `7  ,MI    "Y P'   MM   `7
#        VM:   ,V  ,V^MM.      MM           AW       MM        MM   d    `MMb.          MM
#         MM.  M' ,M  `MM      MM          ,M'       MM        MMmmMM      `YMMNq.      MM
#         `MM A'  AbmmmqMA     MM      ,   MV        MM        MM   Y  , .     `MM      MM
#          :MM;  A'     VML    MM     ,M  AW         MM        MM     ,M Mb     dM      MM
#           VF .AMA.   .AMMA..JMMmmmmMMM ,M'       .JMML.    .JMMmmmmMMM P"Ybmmd"     .JMML.
#                                        MV
#                                       AW



def splitValTest(x_valTest,finalSize,imgSize,count):

    arr_a_list = []
    arr_s_list = []
    arr_c_list = []

    # loop through each patient.
    for arr in iter(x_valTest):

        # offset array to get a smaller one
        offsetArr = offsetPatch(arr, finalSize)

        # get miniPatch (0~9)
        # randInt = random.randint(0,9)
        miniPatch = getMiniPatch(4,offsetArr,imgSize)

        # reshape to make channel
        miniPatch = miniPatch.reshape(imgSize,imgSize,imgSize,1)


        # EXTRACT ORIENTATION SLICES
        skip = 4 # in mm or pixel
        travel = int(count * skip)
        mid  = int(imgSize/2.0)
        #
        arr_a_list.append( miniPatch [(mid-travel):(mid+travel+1):skip,:,:] )
        #
        arr_s = miniPatch [:,:,(mid-travel):(mid+travel+1):skip]
        arr_s_list.append( np.swapaxes( np.rot90(arr_s,3) , 0,2).reshape(count*2+1,imgSize,imgSize,1)  )
        #
        arr_c = miniPatch [:,(mid-travel):(mid+travel+1):skip,:]
        arr_c_list.append( np.swapaxes(np.flipud (arr_c) ,0,1).reshape(count*2+1,imgSize,imgSize,1) )


        
    arr_a_list = np.array( arr_a_list ) 
    arr_s_list = np.array( arr_s_list )
    arr_c_list = np.array( arr_c_list )  
        
    
        
    return arr_a_list,arr_s_list,arr_c_list