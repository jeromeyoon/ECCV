import numpy as np
import random
import cv2
import os
import itertools
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils
from keras.optimizers import SGD
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
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer=fast_run,nvcc.fastmath=True python eccv_label.py '''

savepath = '/research1/YOON/ECCV2016/keras/result/224x224/'


"""
trainednet = glob.glob(savepath+'*hdf5')
if len(trainednet):
    trainednet.reverse()
    model.load_weights(trainednet[0])
"""

nb_epoch = 30

def VGG_16():
   batch_size = 30
   model=makenetwork.vgg_label()
   #sharednet,model1,model2,model3 = makenetwork.eccvmodel_label()
   sgd =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
   model.compile(optimizer='sgd',loss={'out':'categorical_crossentropy'})
   
   h5trainpath = '/research1/YOON/ECCV2016/224x224/h5_train/'
   h5files = glob.glob(h5trainpath+'*.h5')
   h5files.sort()
   nbh5files = len(h5files)

   h5valpath = '/research1/YOON/ECCV2016/224x224/h5_test/'
   h5valfiles = glob.glob(h5valpath+'*.h5')
   h5valfiles.sort()
   nbh5valfiles = len(h5valfiles)
   for idx in range(nbh5valfiles):
       f =h5py.File(h5valfiles[idx],'r')
       if idx == 0:
           input1_val = f['data1'][()]
           input2_val = f['data2'][()]
           label_val = f['label'][()]        
           label_val=label_val.astype('int32')	
	   datasize = input1_val.shape[0]
           label_val = np.reshape(label_val,[datasize])
	   label_val = np_utils.to_categorical(label_val, 13)
           f.close()
       else:
	   tmpinput1 = f['data1'][()]
           tmpinput2 = f['data2'][()]
           tmplabel = f['label'][()]        
           tmplabel=label.astype('int32')	
	   datasize = tmpinput1.shape[0]
           tmplabel = np.reshape(tmplabel,[datasize])
	   tmplabel = np_utils.to_categorical(tmplabel, 13)
	   input1_val=np.concatenate((input1_val,tmpinput1),axis=0)
           input2_val=np.concatenate((input2_val,tmpinput2),axis=0)
           label_val=np.concatenate((label_val,tmplabel),axis=0)
	   f.close()
 
   for iter in range(nb_epoch):
       trainoder = np.random.permutation(nbh5files)
       for i in range(int(nbh5files)):
	   f =h5py.File(h5files[i],'r')
	   input1 = f['data1'][()]
           input2 = f['data2'][()]
           label = f['label'][()]        
           label=label.astype('int32')	
           			
	   datasize = input1.shape[0]
           label = np.reshape(label,[datasize])
	   label = np_utils.to_categorical(label, 13)
           f.close()

           #ouput1_1 =sharednet.fit(input1,label,batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=0)
           #output1_2model1.fit(output1_1,label,batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=0)
           #ouput2_1 =sharednet.fit(input2,label,batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=0)
           #output2_2model1.fit(output2_1,label,batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=0)
           
                    


    	   #model3.fit(model1(sharednet(input1)),label,batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=0)
    	   model.fit({'input1':input1,'input2':input2,'out':label},batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=1)
           
           #trainout = model.predict({'input1':input1,'input2':input2,'out':label},batch_size=batch_size,verbose=0)
           #trainout =  trainout['out']

	   #print('trainerr shape',trainout.shape)
	
       #val_accuracy= model.evaluate({'input1':input1_val,'input2':input2_val,'out':label_val},batch_size=batch_size,verbose=1)      
       #print('val acc:',val_accuracy) 
       #savename = "%05d" % iter
       #model.save_weights(savepath+'model_'+savename+'.hdf5', overwrite=True)
       gc.collect()
   




def vgg_224():
   batch_size =20
   print('VGG 224')
   print('make network and compile')
   model = makenetwork.sharednet_label()
   print('load network')
   sgd =SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) 
   model.compile(optimizer='sgd',loss={'out':'categorical_crossentropy'} )
   
   h5trainpath = '/research1/YOON/ECCV2016/42x42/h5_train/'
   h5files = glob.glob(h5trainpath+'*.h5')
   h5files.sort()
   nbh5files = len(h5files)

   h5valpath = '/research1/YOON/ECCV2016/42x42/h5_test/'
   h5valfiles = glob.glob(h5valpath+'*.h5')
   h5valfiles.sort()
   nbh5valfiles = len(h5valfiles)

   for idx in range(nbh5valfiles):
       f =h5py.File(h5valfiles[idx],'r')
       if idx == 0:
           input1_val = f['data1'][()]
           input2_val = f['data2'][()]
           label_val = f['label'][()]        
			
	   datasize = input1_val.shape[0]
           label_val = np.reshape(label_val,[datasize])
	   label_val = np_utils.to_categorical(label_val, 13)
           f.close()
       else:
	   tmpinput1 = f['data1'][()]
           tmpinput2 = f['data2'][()]
           tmplabel = f['label'][()]        
	   datasize = tmpinput1.shape[0]
           tmplabel = np.reshape(tmplabel,[datasize])
	   tmplabel = np_utils.to_categorical(tmplabel, 13)
	   input1_val=np.concatenate((input1_val,tmpinput1),axis=0)
           input2_val=np.concatenate((input2_val,tmpinput2),axis=0)
           label_val=np.concatenate((label_val,tmplabel),axis=0)
	   f.close()
 
   for iter in range(nb_epoch):
       trainoder = np.random.permutation(nbh5files)
       for i in range(int(nbh5files)):
	   f =h5py.File(h5files[i],'r')
	   input1 = f['data1'][()]
           input2 = f['data2'][()]
           label = f['label'][()]        
    	   input1 = input1.astype('float32')
           input2 = input2.astype('float32')     
         			
	   datasize = input1.shape[0]
           label = np.reshape(label,[datasize])
	   label = np_utils.to_categorical(label, 13)
           f.close()

    	   model.fit({'input1':input1,'input2':input2,'out':label},batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=1)
	
       val_accuracy= model.evaluate({'input1':input1_val,'input2':input2_val,'out':label_val},batch_size=batch_size,verbose=1)      
       print('val acc:',val_accuracy) 
       savename = "%05d" % iter
       model.save_weights(savepath+'model_'+savename+'.hdf5', overwrite=True)
       gc.collect()

def small_vgg():
    for iteration in range(10):
        print('\n'+str(iteration)+'th epoch '+'-'*50)
        progbar = Progbar(target=(10620))
        randfiles = np.random.permutation(nbh5files)   
        for idx in range(nbh5files):    
            f =h5py.File(h5files[idx],'r')
	    if idx == 0:
	        input1 = f['data1'][()]
                input2 = f['data2'][()]
                label = f['label'][()]        
    	        input1 = input1.astype('float32')
                input2 = input2.astype('float32')     
         			
	        datasize = input1.shape[0]
                label = np.reshape(label,[datasize])
	        label = np_utils.to_categorical(label, 13)
                f.close()
	   
	    else:
	        tmpinput1 = f['data1'][()]
                tmpinput2 = f['data2'][()]
                tmplabel = f['label'][()]        
	        datasize = tmpinput1.shape[0]
    	        tmpinput1 = tmpinput1.astype('float32')
                tmpinput2 = tmpinput2.astype('float32')     
                tmplabel = np.reshape(tmplabel,[datasize])
	        tmplabel = np_utils.to_categorical(tmplabel, 13)
	        input1=np.concatenate((input1,tmpinput1),axis=0)
                input2=np.concatenate((input2,tmpinput2),axis=0)
                label=np.concatenate((label,tmplabel),axis=0)
	        f.close()
        for idx in range(nbh5valfiles):    
            f =h5py.File(h5valfiles[idx],'r')
	    if idx == 0:
	        input1_val = f['data1'][()]
                input2_val = f['data2'][()]
                label_val = f['label'][()]        
			
	        datasize = input1_val.shape[0]
                label_val = np.reshape(label_val,[datasize])
	        label_val = np_utils.to_categorical(label_val, 13)
                f.close()
	    else:
	        tmpinput1 = f['data1'][()]
                tmpinput2 = f['data2'][()]
                tmplabel = f['label'][()]        
	        datasize = tmpinput1.shape[0]
                tmplabel = np.reshape(tmplabel,[datasize])
	        tmplabel = np_utils.to_categorical(tmplabel, 13)
	        input1_val=np.concatenate((input1_val,tmpinput1),axis=0)
                input2_val=np.concatenate((input2_val,tmpinput2),axis=0)
                label_val=np.concatenate((label_val,tmplabel),axis=0)
	        f.close()
 
        traindata_size = input1.shape[0]
        valdata_size = input1_val.shape[0] 
        nb_trainbatch = int(np.ceil(float(traindata_size)/batch_size))

        model.fit({'input1':input1,'input2':input2,'out':label},batch_size=10,nb_epoch=1,shuffle=False,verbose=1,validation_data={'input1':input1_val,'input2':input2_val,'out':label_val})
        val_accuracy= model.evaluate({'input1':input1_val,'input2':input2_val,'out':label_val},batch_size=batch_size,verbose=1)      
        print('val acc:',val_accuracy) 
        ## save model weights
        savename = "%05d" % iter
        model.save_weights(savepath+'model_'+savename+'.hdf5', overwrite=True)
        gc.collect()
   
print('starting') 
VGG_16() 
