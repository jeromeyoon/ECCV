import numpy as np
import random
import cv2
import os
import itertools
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils
import makenetwork
import keras.callbacks
import pdb
import os.path
import glob
import time
import sys
import gc
#import tensorflow
import h5py
from six.moves import cPickle

''' THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer=fast_run,nvcc.fastmath=True python eccv_label_test.py '''
def loadnet(trainednet= None):
  model = makenetwork.siames_label()
  if trainednet:
   	model.load_weights(trainednet) 
  return model


if __name__ =="__main__":

  trainednet = '/research1/YOON/ECCV2016/keras/result/model_00000.hdf5'
  
  LFpath ='/research1/db/iccv2015/HCI/train/'
  h5files = glob.glob(LFpath+'*.h5')
  h5files.sort()
  

  print('choose a LF image:')
  LF_img =input()
  print('the first sub-aperture:(ex:cols,rows)')
  sub_first = input()
  print('the second sub-aperture:(ex:cols,rows)')
  sub_second = input()

  
 
  if sub_first == sub_second or LF_img >= len(h5files) or len(sub_first) != 2 or len(sub_second) !=2:
	print(' Error choose LF images carefully')
  else:
  	print('load LF image')
	f = h5py.File(h5files[LF_img],'r')
	images = f['LF'][()]
	#images = np.transpose(images,(4,3,2,1,0)) 
 	
	sub1_img =images[sub_first[0],sub_first[1],:,:,:]  
	sub2_img =images[sub_second[0],sub_second[1],:,:,:]  
	
	sub1_img = cv2.cvtColor(sub1_img,cv2.COLOR_BGR2GRAY)
	sub2_img = cv2.cvtColor(sub2_img,cv2.COLOR_BGR2GRAY)
        sub1_img = np.reshape(sub1_img,(1,1,sub1_img.shape[0],sub1_img.shape[1]))
        sub2_img = np.reshape(sub2_img,(1,1,sub2_img.shape[0],sub2_img.shape[1]))
	print('images shape:',images.shape)
        print('sub1 shape:',sub1_img.shape)
        print('sub2 shape:',sub2_img.shape)   	
	
	print('load trained network')
	model = loadnet(trainednet)
	model.compile(optimizer='sgd',loss={'out':'categorical_crossentropy'} )
	out =model.predict({'input1':sub1_img,'input2':sub2_img})
	print np.argmax(out)
	
	
