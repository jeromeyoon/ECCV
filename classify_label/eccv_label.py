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
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer=fast_run,nvcc.fastmath=True python eccv_label.py '''

savepath = '/research1/YOON/ECCV2016/keras/result/'

print('make network and compile')
model = makenetwork.siames_label()
print('load network')
model.compile(optimizer='adam',loss={'out':'categorical_crossentropy'} )
print('End compiling model')

trainednet = glob.glob(savepath+'*hdf5')
if len(trainednet):
    trainednet.reverse()
    model.load_weights(trainednet[0])


h5trainpath = '/research1/YOON/ECCV2016/42x42/h5_train/'
h5files = glob.glob(h5trainpath+'*.h5')
h5files.sort()
nbh5files = len(h5files)

h5valpath = '/research1/YOON/ECCV2016/42x42/h5_test/'
h5valfiles = glob.glob(h5valpath+'*.h5')
h5valfiles.sort()
nbh5valfiles = len(h5valfiles)

nb_epoch = 30
batch_size = 128
for iteration in range(1):
    # datapath='/media/disk1/bgsim/Dataset/UCF-101'
    # trainlist,testlist = makeDB(datapath=datapath,divideself=False)
    
    # np.random.shuffle(namesord)
    
    seen = 0
    totloss = 0
    totacc = 0
    apseen = 0
    totobjAP = 0
    totactAP = 0
    timestepactAP = 0
    lastacc = 0
    print('\n'+str(iteration)+'th epoch '+'-'*50)
    
    progbar = Progbar(target=(10620))
    
    passed = 0
    randfiles = np.random.permutation(nbh5files)   
    for idx in range(nbh5files):
    # for xxx in range(1):    
        
        f =h5py.File(h5files[idx],'r')
	if idx == 0:
	   input1 = f['data1'][()]
           input2 = f['data2'][()]
           label = f['label'][()]        
         			
	   datasize = input1.shape[0]
           label = np.reshape(label,[datasize])
	   label = np_utils.to_categorical(label, 13)
           f.close()
	else:
	   tmpinput1 = f['data1'][()]
           tmpinput2 = f['data2'][()]
           tmplabel = f['label'][()]        
	   datasize = tmpinput1.shape[0]
           tmplabel = np.reshape(tmplabel,[datasize])
	   tmplabel = np_utils.to_categorical(tmplabel, 13)
	   input1=np.concatenate((input1,tmpinput1),axis=0)
           input2=np.concatenate((input2,tmpinput2),axis=0)
           label=np.concatenate((label,tmplabel),axis=0)
	   f.close()
    input1 = input1.astype('float32')
    input2 = input2.astype('float32')     
 
    for idx in range(nbh5valfiles):
    # for xxx in range(1):    
        
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
    for iter in range(nb_epoch):
        for batch_iter in range(int(nb_trainbatch)):
	    print('-'*30)
	    print('batch %d/%d') % (batch_iter, nb_trainbatch)
	    batch_end =(batch_iter+1) * batch_size
	    if batch_end > input1.shape[0]:
		nb_samples = input1.shape[0] - batch_iter * batch_size
	    else:
                nb_samples = batch_size

            loss=model.train_on_batch({'input1':input1[batch_iter*batch_size:batch_iter*batch_size+nb_samples],'input2':input2[batch_iter*batch_size:(batch_iter)*batch_size+nb_samples],'out':label[batch_iter*batch_size:(batch_iter)*batch_size+nb_samples]})
            #loss=model.train_on_batch({'input1':input1[batch_iter*batch_size:(batch_iter+1)*batch_size],'input2':input2[batch_iter*batch_size:(batch_iter+1)*batch_size],'out':label[batch_iter*batch_size:(batch_iter+1)*batch_size]})
	    print('train loss:',loss[0])
	    #model.fit({'input1':input1[batch_iter*batch_size:(batch_iter+1)*batch_size],'input2':input2[batch_iter*batch_size:(batch_iter+1)*batch_size],'out':label[batch_iter*batch_size:(batch_iter+1)*batch_size]},batch_size=batch_size,nb_epoch=1,shuffle=False,verbose=1)
           # print(histoty.hist)

    #model.fit({'input1':input1,'input2':input2,'out':label},batch_size=10,nb_epoch=nb_epoch,shuffle=False,verbose=1,validation_data={'input1':input1_val,'input2':input2_val,'out':label_val})
    #endfit = time.time()
   
        train_accuracy= model.evaluate({'input1':input1,'input2':input2,'out':label},batch_size=batch_size,verbose=1)
        print('train acc:',train_accuracy)      
        val_accuracy= model.evaluate({'input1':input1_val,'input2':input2_val,'out':label_val},batch_size=batch_size,verbose=1)      
        print('val acc:',val_accuracy) 
    	### save model weights
	savename = "%05d" % iter
        model.save_weights(savepath+'model_'+savename+'.hdf5', overwrite=True)
        #progbar.update(tidx+passed)
    
    	gc.collect()
    
