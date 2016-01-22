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
import scipy.io
import getAP
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_run,nvcc.fastmath=True python eccv_label_test.py '''

def loadnet(trainednet= None):
  model = makenetwork.sharednet_label_dropout()
  if trainednet:
   	model.load_weights(trainednet)
	print('load pretrained network:',trainednet) 
  return model


def gridposition(matfile):
   numcom = matfile.shape[0]
   count =0
   for k in range(numcom):
       first = matfile[k][0]
       second = matfile[k][1]
       g = matfile[k][5]
       numcombination = first.shape[1]
       print('com1 %d com2 %d') % (k,numcombination)
       for m in range(numcombination):
           if count ==0:
               firstpt = first[0][m]
               secondpt = second[0][m]
               gt = g
               count +=1
           else: 
               firstpt = np.concatenate((firstpt,first[0][m]),axis=0)
               secondpt= np.concatenate((secondpt,second[0][m]),axis=0)
	       gt = np.concatenate((gt,g),axis=0)	
   return firstpt,secondpt,gt

if __name__ =="__main__":

  trainednet = '/research1/YOON/ECCV2016/keras/result/42x42_dropout/model_00079.hdf5'
  #combination = 'LF_position.mat'
  #matfile = scipy.io.loadmat(combination)
  #firstpt,secondpt,gt = gridposition(matfile['result'])
  #firstpt -=1
  #secondpt -=1
  
  h5trainpath = '/research1/YOON/ECCV2016/42x42_2/h5_train/'
  h5files = glob.glob(h5trainpath+'*.h5')
  h5files.sort()
  nbh5files = len(h5files)
  
  print('choose a LF image:0~%d') % nbh5files
  LF_img =input()

  for idx in range(nbh5files):
       f =h5py.File(h5files[LF_img],'r')
       input1 = f['data1'][()]
       input2 = f['data2'][()]
       label = f['label'][()]        
       label=label.astype('int32')	
       datasize = input1.shape[0]
       label = np.reshape(label,[datasize])
       label = np_utils.to_categorical(label, 13)
       f.close()



  if LF_img >= len(h5files):
  #if sub_first == sub_second or LF_img >= len(h5files) or len(sub_first) != 2 or len(sub_second) !=2:
	print(' Error choose LF image carefully')
  else:
  	print('load LF image')
        ap =0.0; 
	model = loadnet(trainednet)
	model.compile(optimizer='sgd',loss={'out':'categorical_crossentropy'} )
	out =model.predict({'input1':input1,'input2':input2})
        out = np.argmax(out['out'],axis=-1)
        acc =getAP.loss(out,label) 
        print('acc:',acc)
