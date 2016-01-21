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

  gc.collect()
  trainednet = '/research1/YOON/ECCV2016/keras/result/42x42_dropout/model_00040.hdf5'
  combination = 'LF_position.mat'
  matfile = scipy.io.loadmat(combination)
  firstpt,secondpt,gt = gridposition(matfile['result'])
  firstpt -=1
  secondpt -=1
  print('point shape:',secondpt.shape) 
  LFpath ='/research1/db/iccv2015/HCI/train/'
  h5files = glob.glob(LFpath+'*.h5')
  h5files.sort()
  

  print('choose a LF image:')
  LF_img =input()
  """
  print('the first sub-aperture:(ex:cols,rows)')
  sub_first = input()
  print('the second sub-aperture:(ex:cols,rows)')
  sub_second = input()
  """
  """
  im  = cv2.imread('./moon.jpg')
  im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  print('image shape :',im_gray.shape)
  cv2.imshow('moon',im_gray)
  cv2.waitKey()
  """
  """
  h5trainpath = '/research1/YOON/ECCV2016/42x42_2/h5_train/'
  h5files = glob.glob(h5trainpath+'*.h5')
  h5files.sort()
  nbh5files = len(h5files)
  f =h5py.File(h5files[0],'r')
  input1 = f['data1'][()]
  input2 = f['data2'][()]
  label = f['label'][()]        
  label=label.astype('int32')	
  datasize = input1.shape[0]
  label = np.reshape(label,[datasize])
  label = np_utils.to_categorical(label, 13)
  f.close()
  """


  if LF_img >= len(h5files):
  #if sub_first == sub_second or LF_img >= len(h5files) or len(sub_first) != 2 or len(sub_second) !=2:
	print(' Error choose LF image carefully')
  else:
  	print('load LF image')
	f = h5py.File(h5files[LF_img],'r')
	images = f['LF'][()]
	#print('loaded LF image:',h5files[LF_img])
        #print('LF img shape :',images.shape)
        #images = np.transpose(images,(1,0,2,3,4)) 
        ap =0.0; 
        numcom = firstpt.shape[0]
	model = loadnet(trainednet)
	model.compile(optimizer='sgd',loss={'out':'categorical_crossentropy'} )
        """
        print('input1 shape:',input1.shape)
        print('input1 type:',type(input1))
        sub1_img = input1[10,:,:,:]
        sub2_img = input2[10,:,:,:]
        sub1_img = np.expand_dims(sub1_img,axis=0)
        sub2_img = np.expand_dims(sub2_img,axis=0)
        
        print('gt :',np.argmax(label[10])) 
        """   
        
        
        """ 
        cv2.imshow('img1',sub1_img)
        cv2.imshow('img2',sub2_img)
        cv2.waitKey()
        """
        """	
        sub1_img = cv2.cvtColor(sub1_img,cv2.COLOR_BGR2GRAY)
	sub2_img = cv2.cvtColor(sub2_img,cv2.COLOR_BGR2GRAY)
	sub1_img = sub1_img[10:52,10:52]
	sub2_img = sub2_img[10:52,10:52]
	sub1_img = sub1_img.astype('float32')
	sub2_img = sub2_img.astype('float32')
	sub1_img /=255
	sub2_img /=255
        sub1_img = np.expand_dims(sub1_img,axis=0)
        sub1_img = np.expand_dims(sub1_img,axis=0)
        sub2_img = np.expand_dims(sub2_img,axis=0)
        sub2_img = np.expand_dims(sub2_img,axis=0)
	out =model.predict({'input1':sub1_img,'input2':sub2_img})
        #print ('out shape:',out.shape)
        print np.argmax(out['out'])
        """
	saveim = True
        for id in range(numcom): 	
	    sub1_img =images[firstpt[id][0],firstpt[id][1],:,:,:]  
	    sub2_img =images[secondpt[id][0],secondpt[id][1],:,:,:] 
            if saveim:
	        cv2.imwrite('img1.jpg',sub1_img)
                cv2.imwrite('im2.jpg',sub2_img)
	        saveim =False

	    sub1_img = cv2.cvtColor(sub1_img,cv2.COLOR_BGR2GRAY)
	    sub2_img = cv2.cvtColor(sub2_img,cv2.COLOR_BGR2GRAY)
	    sub1_img = sub1_img[10:52,10:52]
	    sub2_img = sub2_img[10:52,10:52]
	    sub1_img = sub1_img.astype('float32')
	    sub2_img = sub2_img.astype('float32')
	    sub1_img /=255
	    sub2_img /=255
            sub1_img = np.expand_dims(sub1_img,axis=0)
            sub1_img = np.expand_dims(sub1_img,axis=0)
            sub2_img = np.expand_dims(sub2_img,axis=0)
            sub2_img = np.expand_dims(sub2_img,axis=0)
	
	    #sub1_img = np.reshape(sub1_img,(1,1,sub1_img.shape[0],sub1_img.shape[1]))
            #sub2_img = np.reshape(sub2_img,(1,1,sub2_img.shape[0],sub2_img.shape[1]))
            #print('images shape:',images.shape)
            #print('sub1 shape:',sub1_img.shape)
            #print('sub2 shape:',sub2_img.shape)   	
	
	    out =model.predict({'input1':sub1_img,'input2':sub2_img})
	    #print('output:',out)
	    #print np.argmax(out['out'])
	    if gt[id][0] == np.argmax(out['out']):
	        ap +=1
        ap/=numcom	
	print(' AP :%f') % ap
