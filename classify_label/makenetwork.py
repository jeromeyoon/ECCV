from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, MaskedLayer, Flatten, RepeatVector, Reshape,Siamese,Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras import backend as K
import copy
import theano
import pdb


# merge after dropout? or merge before dropout?
# merge_mode : 'sum' or 'concat'

def sharednet_label_dropout():
    model = Graph()
    model.add_input(name='input1',input_shape=(1,42,42))
    model.add_input(name='input2',input_shape=(1,42,42))
    
    conv1_1 = Convolution2D(64,3,3,activation='relu',border_mode='same') 
    conv1_2 = copy.deepcopy(conv1_1) 
    model.add_node(conv1_1,name='conv1_1',input='input1')    
    model.add_node(conv1_2,name='conv1_2',input='input2')    
    model.nodes['conv1_2'].W = model.nodes['conv1_1'].W
    model.nodes['conv1_2'].b = model.nodes['conv1_1'].b
    model.nodes['conv1_2'].params =[]
    conv2_1 = Convolution2D(64,3,3,activation='relu',border_mode='same') 
    conv2_2 = copy.deepcopy(conv2_1)
    model.add_node(conv2_1,name='conv2_1',input='conv1_1')    
    model.add_node(conv2_2,name='conv2_2',input='conv1_2')    
    model.nodes['conv2_2'].W = model.nodes['conv2_1'].W
    model.nodes['conv2_2'].b = model.nodes['conv2_1'].b
    model.nodes['conv2_2'].params =[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_1',input='conv2_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_2',input='conv2_2')

    conv3_1 = Convolution2D(128,3,3,activation='relu',border_mode='same') 
    conv3_2 = copy.deepcopy(conv3_1)
    model.add_node(conv3_1,name='conv3_1',input='pool2_1')    
    model.add_node(conv3_2,name='conv3_2',input='pool2_2')    
    model.nodes['conv3_2'].W = model.nodes['conv3_1'].W
    model.nodes['conv3_2'].b = model.nodes['conv3_1'].b
    model.nodes['conv3_2'].params=[]
    conv4_1 = Convolution2D(128,3,3,activation='relu',border_mode='same') 
    conv4_2 = copy.deepcopy(conv4_1)
    model.add_node(conv4_1,name='conv4_1',input='conv3_1')    
    model.add_node(conv4_2,name='conv4_2',input='conv3_2')    
    model.nodes['conv4_2'].W = model.nodes['conv4_1'].W
    model.nodes['conv4_2'].b = model.nodes['conv4_1'].b
    model.nodes['conv4_2'].params=[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool4_1',input='conv4_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool4_2',input='conv4_2')


    conv5_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv5_2 = copy.deepcopy(conv5_1)
    model.add_node(conv5_1,name='conv5_1',input='pool4_1')    
    model.add_node(conv5_2,name='conv5_2',input='pool4_2')    
    model.nodes['conv5_2'].W = model.nodes['conv5_1'].W
    model.nodes['conv5_2'].b = model.nodes['conv5_1'].b
    model.nodes['conv5_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat5_1',input='conv5_1')
    #model.add_node(BatchNormalization(),name='Bat5_2',input='conv5_2')

    conv6_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv6_2 = copy.deepcopy(conv6_1)
    model.add_node(conv6_1,name='conv6_1',input='conv5_1')    
    model.add_node(conv6_2,name='conv6_2',input='conv5_2')    
    model.nodes['conv6_2'].W = model.nodes['conv6_1'].W
    model.nodes['conv6_2'].b = model.nodes['conv6_1'].b
    model.nodes['conv6_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat6_1',input='conv6_1')
    #model.add_node(BatchNormalization(),name='Bat6_2',input='conv6_2')
    conv7_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv7_2 = copy.deepcopy(conv7_1)
    model.add_node(conv7_1,name='conv7_1',input='conv6_1')    
    model.add_node(conv7_2,name='conv7_2',input='conv6_2')    
    model.nodes['conv7_2'].W = model.nodes['conv7_1'].W
    model.nodes['conv7_2'].b = model.nodes['conv7_1'].b
    model.nodes['conv7_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat7_1',input='conv7_1')
    #model.add_node(BatchNormalization(),name='Bat7_2',input='conv7_2')

    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool7_1',input='conv7_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool7_2',input='conv7_2')

    model.add_node(Flatten(),name='Flat8_1',input='pool7_1')
    model.add_node(Flatten(),name='Flat8_2',input='pool7_2')


    dense8_1 = Dense(1024,activation='relu')
    dense8_2 = copy.deepcopy(dense8_1)
    model.add_node(dense8_1,name='dense8_1',input='Flat8_1')
    model.add_node(dense8_2,name='dense8_2',input='Flat8_2')
    model.nodes['dense8_2'].W = model.nodes['dense8_1'].W
    model.nodes['dense8_2'].b = model.nodes['dense8_1'].b
    model.nodes['dense8_2'].params =[]

    dense9 = Dense(1024,activation='relu')
    model.add_node(dense9,name='dense9',inputs=['dense8_1','dense8_2'],merge_mode='sum')
    model.add_node(Dropout(0.25), name = 'dropout10',input='dense9')
    dense10 = Dense(1024,activation='relu')
    model.add_node(dense10,name='dense11',input='dropout10')
    model.add_node(Dropout(0.25),name='dropout12',input='dense11')
    model.add_node(Dense(13),name='dense13',input='dropout12')
    model.add_node(Activation('softmax'),name='softmax',input='dense13')
    model.add_output(name='out',input='softmax')

    return model


def sharednet_label():
    model = Graph()
    model.add_input(name='input1',input_shape=(1,42,42))
    model.add_input(name='input2',input_shape=(1,42,42))
    
    conv1_1 = Convolution2D(64,3,3,activation='relu',border_mode='same') 
    conv1_2 = copy.deepcopy(conv1_1) 
    model.add_node(conv1_1,name='conv1_1',input='input1')    
    model.add_node(conv1_2,name='conv1_2',input='input2')    
    model.nodes['conv1_2'].W = model.nodes['conv1_1'].W
    model.nodes['conv1_2'].b = model.nodes['conv1_1'].b
    model.nodes['conv1_2'].params =[]
    conv2_1 = Convolution2D(64,3,3,activation='relu',border_mode='same') 
    conv2_2 = copy.deepcopy(conv2_1)
    model.add_node(conv2_1,name='conv2_1',input='conv1_1')    
    model.add_node(conv2_2,name='conv2_2',input='conv1_2')    
    model.nodes['conv2_2'].W = model.nodes['conv2_1'].W
    model.nodes['conv2_2'].b = model.nodes['conv2_1'].b
    model.nodes['conv2_2'].params =[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_1',input='conv2_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_2',input='conv2_2')

    conv3_1 = Convolution2D(128,3,3,activation='relu',border_mode='same') 
    conv3_2 = copy.deepcopy(conv3_1)
    model.add_node(conv3_1,name='conv3_1',input='pool2_1')    
    model.add_node(conv3_2,name='conv3_2',input='pool2_2')    
    model.nodes['conv3_2'].W = model.nodes['conv3_1'].W
    model.nodes['conv3_2'].b = model.nodes['conv3_1'].b
    model.nodes['conv3_2'].params=[]
    conv4_1 = Convolution2D(128,3,3,activation='relu',border_mode='same') 
    conv4_2 = copy.deepcopy(conv4_1)
    model.add_node(conv4_1,name='conv4_1',input='conv3_1')    
    model.add_node(conv4_2,name='conv4_2',input='conv3_2')    
    model.nodes['conv4_2'].W = model.nodes['conv4_1'].W
    model.nodes['conv4_2'].b = model.nodes['conv4_1'].b
    model.nodes['conv4_2'].params=[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool4_1',input='conv4_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool4_2',input='conv4_2')


    conv5_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv5_2 = copy.deepcopy(conv5_1)
    model.add_node(conv5_1,name='conv5_1',input='pool4_1')    
    model.add_node(conv5_2,name='conv5_2',input='pool4_2')    
    model.nodes['conv5_2'].W = model.nodes['conv5_1'].W
    model.nodes['conv5_2'].b = model.nodes['conv5_1'].b
    model.nodes['conv5_2'].params=[]
    conv6_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv6_2 = copy.deepcopy(conv6_1)
    model.add_node(conv6_1,name='conv6_1',input='conv5_1')    
    model.add_node(conv6_2,name='conv6_2',input='conv5_2')    
    model.nodes['conv6_2'].W = model.nodes['conv6_1'].W
    model.nodes['conv6_2'].b = model.nodes['conv6_1'].b
    model.nodes['conv6_2'].params=[]
    conv7_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv7_2 = copy.deepcopy(conv7_1)
    model.add_node(conv7_1,name='conv7_1',input='conv6_1')    
    model.add_node(conv7_2,name='conv7_2',input='conv6_2')    
    model.nodes['conv7_2'].W = model.nodes['conv7_1'].W
    model.nodes['conv7_2'].b = model.nodes['conv7_1'].b
    model.nodes['conv7_2'].params=[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool7_1',input='conv7_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool7_2',input='conv7_2')

    model.add_node(Flatten(),name='Flat8_1',input='pool7_1')
    model.add_node(Flatten(),name='Flat8_2',input='pool7_2')


    dense8_1 = Dense(1024,activation='relu')
    dense8_2 = copy.deepcopy(dense8_1)
    model.add_node(dense8_1,name='dense8_1',input='Flat8_1')
    model.add_node(dense8_2,name='dense8_2',input='Flat8_2')
    model.nodes['dense8_2'].W = model.nodes['dense8_1'].W
    model.nodes['dense8_2'].b = model.nodes['dense8_1'].b
    model.nodes['dense8_2'].params =[]

    dense9 = Dense(1024,activation='relu')
    model.add_node(dense9,name='dense9',inputs=['dense8_1','dense8_2'],merge_mode='sum')
    dense10 = Dense(1024,activation='relu')
    model.add_node(dense10,name='dense10',input='dense9')
    model.add_node(Dense(13),name='dense11',input='dense10')
    model.add_node(Activation('softmax'),name='softmax',input='dense11')
    model.add_output(name='out',input='softmax')

    return model
    
def siames_label_vgg():
    model = Graph()
    model.add_input(name='input1',input_shape=(1,224,224))
    model.add_input(name='input2', input_shape=(1,224,224))
    shared_conv1_1  = Convolution2D(64,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv1_1,name='shared_conv1_1',inputs=['input1','input2'],outputs=['output1_conv1_1','output2_conv1_1'])
    
    shared_conv1_2 = Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv1_2,name='shared_conv1_2',inputs= ['output1_conv1_1','output2_conv1_1'],outputs=['output1_conv1_2','output2_conv1_2'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_1',input='output1_conv1_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_2',input='output2_conv1_2')
    
    shared_conv2_1= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv2_1,name='shared_conv2_1',inputs=['pool1_1','pool1_2'],outputs=['output1_conv2_1','output2_conv2_1'])
    shared_conv2_2= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv2_2,name='shared_conv2_2',inputs=['output1_conv2_1','output2_conv2_1'],outputs=['output1_conv2_2','output2_conv2_2'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_1',input='output1_conv2_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_2',input='output2_conv2_2')
    
    shared_conv3_1= Convolution2D(256,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_1,name='shared_conv3_1',inputs=['pool2_1','pool2_2'],outputs=['output1_conv3_1','output2_conv3_1'])
    shared_conv3_2= Convolution2D(256,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_2,name='shared_conv3_2',inputs=['output1_conv3_1','output2_conv3_1'],outputs=['output1_conv3_2','output2_conv3_2'])
    shared_conv3_3= Convolution2D(256,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_3,name='shared_conv3_3',inputs=['output1_conv3_2','output2_conv3_2'],outputs=['output1_conv3_3','output2_conv3_3'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool3_1',input='output1_conv3_3')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool3_2',input='output2_conv3_3')

    shared_conv4_1= Convolution2D(512,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv4_1,name='shared_conv4_1',inputs=['pool3_1','pool3_2'],outputs=['output1_conv4_1','output2_conv4_1'])
    shared_conv4_2= Convolution2D(512,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv4_2,name='shared_conv4_2',inputs=['output1_conv4_1','output2_conv4_1'],outputs=['output1_conv4_2','output2_conv4_2'])
    shared_conv4_3= Convolution2D(512,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv4_3,name='shared_conv4_3',inputs=['output1_conv4_2','output2_conv4_2'],outputs=['output1_conv4_3','output2_conv4_3'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool4_1',input='output1_conv4_3')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool4_2',input='output2_conv4_3')
 
    shared_conv5_1= Convolution2D(512,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv5_1,name='shared_conv5_1',inputs=['pool4_1','pool4_2'],outputs=['output1_conv5_1','output2_conv5_1'])
    shared_conv5_2= Convolution2D(512,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv5_2,name='shared_conv5_2',inputs=['output1_conv5_1','output2_conv5_1'],outputs=['output1_conv5_2','output2_conv5_2'])
    shared_conv5_3= Convolution2D(512,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv5_3,name='shared_conv5_3',inputs=['output1_conv5_2','output2_conv5_2'],outputs=['output1_conv5_3','output2_conv5_3'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool5_1',input='output1_conv5_3')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool5_2',input='output2_conv5_3')
  
    model.add_shared_node(Flatten(),name='Flat6',inputs=['pool5_1','pool5_2'],outputs=['Flat6_1','Flat6_2'])
    shared_dense6 = Dense(4096,activation='relu')
    model.add_shared_node(shared_dense6,name ='dense6',inputs=['Flat6_1','Flat6_2'],outputs=['dense6_1','dense6_2'])
    model.add_node(Dense(4096,activation='relu'),name='concatlayer',inputs=['dense6_1','dense6_2'],merge_mode='sum')    
     
    model.add_node(Dense(4096,activation='relu'),name='dense7',input='concatlayer')
    model.add_node(Dense(13,activation='relu'),name='dense8',input='dense7')

    model.add_node(Activation('softmax'),name='softmax',input='dense8')
    model.add_output(name='out',input='softmax')

    return model


def siames_label():
    
    model = Graph()
    model.add_input(name='input1',input_shape=(1,42,42))
    model.add_input(name='input2', input_shape=(1,42,42))
    shared_conv1_1  = Convolution2D(64,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv1_1,name='shared_conv1_1',inputs=['input1','input2'],outputs=['output1_conv1_1','output2_conv1_1'])
    
    shared_conv1_2 = Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv1_2,name='shared_conv1_2',inputs= ['output1_conv1_1','output2_conv1_1'],outputs=['output1_conv1_2','output2_conv1_2'])

    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_1',input='output1_conv1_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_2',input='output2_conv1_2')
    
    shared_conv2_1= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv2_1,name='shared_conv2_1',inputs=['pool1_1','pool1_2'],outputs=['output1_conv2_1','output2_conv2_1'])
    
    shared_conv2_2= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv2_2,name='shared_conv2_2',inputs=['output1_conv2_1','output2_conv2_1'],outputs=['output1_conv2_2','output2_conv2_2'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_1',input='output1_conv2_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_2',input='output2_conv2_2')
    
    shared_conv3_1= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_1,name='shared_conv3_1',inputs=['pool2_1','pool2_2'],outputs=['output1_conv3_1','output2_conv3_1'])
    
    shared_conv3_2= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_2,name='shared_conv3_2',inputs=['output1_conv3_1','output2_conv3_1'],outputs=['output1_conv3_2','output2_conv3_2'])
    
    shared_conv3_3= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_3,name='shared_conv3_3',inputs=['output1_conv3_2','output2_conv3_2'],outputs=['output1_conv3_3','output2_conv3_3'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool3_1',input='output1_conv3_3')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool3_2',input='output2_conv3_3')
    
    model.add_shared_node(Flatten(),name='Flat4',inputs=['pool3_1','pool3_2'],outputs=['Flat4_1','Flat4_2'])
    #model.add_node(Flatten(),name='Flat4_2',input='pool3_2')
    
    shared_dense5 = Dense(1024,activation='relu')
    model.add_shared_node(shared_dense5,name ='dense5',inputs=['Flat4_1','Flat4_2'],outputs=['dense5_1','dense5_2'])
    model.add_node(Dense(1024,activation='relu'),name='concatlayer',inputs=['dense5_1','dense5_2'],merge_mode='sum')    
     
    model.add_node(Dense(1024,activation='relu'),name='dense7',input='concatlayer')
    model.add_node(Dense(1024,activation='relu'),name='dense8',input='dense7')
    model.add_node(Dense(13),name='dense9',input='dense8')
    model.add_node(Activation('softmax'),name='softmax',input='dense9')
    model.add_output(name='out',input='softmax')
    return model
    


   


def siames_model():
    model = Graph()
    ### input 1 ####
    model.add_input(name='input1',input_shape=(1,42,42))
    model.add_input(name='input2', input_shape=(1,42,42))
    shared_conv1_1  = Convolution2D(64,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv1_1,name='shared_conv1_1',inputs=['input1','input2'])
    
    shared_conv1_2 = Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv1_2,name='shared_conv1_2',inputs= ['shared_conv1_1'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_1',input='shared_conv1_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_2',input='shared_conv1_2')
    
    shared_conv2_1= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv2_1,name='shared_conv2_1',inputs=['pool1_1','pool1_2'])
    
    shared_conv2_2= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv2_2,name='shared_conv2_2',inputs=['shared_conv2_1'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_1',input='shared_conv2_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_2',input='shared_conv2_2')
    
    shared_conv3_1= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_1,name='shared_conv3_1',inputs=['pool2_1','pool2_2'])
    
    shared_conv3_2= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_2,name='shared_conv3_2',inputs=['shared_conv3_1'])
    
    shared_conv3_3= Convolution2D(128,3,3,activation='relu',border_mode='same')
    model.add_shared_node(shared_conv3_3,name='shared_conv3_3',inputs=['shared_conv3_2'])
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool3_1',input='shared_conv3_3')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool3_2',input='shared_conv3_3')
    
    model.add_node(Flatten(),name='Flat4_1',input='pool3_1')
    model.add_node(Flatten(),name='Flat4_2',input='pool3_2')
    
    shared_dense5 = Dense(512)
    model.add_shared_node(shared_dense5,name ='dense5',inputs=['Flat4_1','Flat4_2'],outputs=['dense5_1','dense5_2'])
    
    #model.add_node(Dense(1024),name='dense6_1',input='dense5_1')
    #model.add_node(Dense(1024),name='dense6_2',input='dense5_2')
    #model.add_node(Dense(512),name='dense5_2',input='Flat4_2')
    #shared_dense6= Dense(1024)
    #model.add_node(shared_dense6,name='dense6',input='dense5',merge_mode='sum')
    #model.add_node(Dense(1024),name='dense6_1',input='dense5_1')
    #model.add_node(Dense(1024),name='dense6_2',input='dense5_2')
    model.add_node(Dense(1024),name='dense6',inputs=['dense5_1','dense5_2'],merge_mode='concat')
    #model.add_node(Dense(1024),name='dense7',input='dense6')
    model.add_node(Dense(13),name='dense7',input='dense6')

    model.add_node(Activation('softmax'),name='softmax',input='dense7')

    model.add_output(name='out',input='softmax')
    return model
    
    


def alex_sharednet_label():
    model = Graph()
    model.add_input(name='input1',input_shape=(1,42,42))
    model.add_input(name='input2',input_shape=(1,42,42))
    
    conv1_1 = Convolution2D(96,11,11,activation='relu',border_mode='same') 
    conv1_2 = copy.deepcopy(conv1_1) 
    model.add_node(conv1_1,name='conv1_1',input='input1')    
    model.add_node(conv1_2,name='conv1_2',input='input2')    
    model.nodes['conv1_2'].W = model.nodes['conv1_1'].W
    model.nodes['conv1_2'].b = model.nodes['conv1_1'].b
    model.nodes['conv1_2'].params =[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_1',input='conv1_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1_2',input='conv1_2')


    conv2_1 = Convolution2D(256,5,5,activation='relu',border_mode='same') 
    conv2_2 = copy.deepcopy(conv2_1)
    model.add_node(conv2_1,name='conv2_1',input='pool1_1')    
    model.add_node(conv2_2,name='conv2_2',input='pool1_2')    
    model.nodes['conv2_2'].W = model.nodes['conv2_1'].W
    model.nodes['conv2_2'].b = model.nodes['conv2_1'].b
    model.nodes['conv2_2'].params =[]
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_1',input='conv2_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2_2',input='conv2_2')

    conv3_1 = Convolution2D(384,3,3,activation='relu',border_mode='same') 
    conv3_2 = copy.deepcopy(conv3_1)
    model.add_node(conv3_1,name='conv3_1',input='pool2_1')    
    model.add_node(conv3_2,name='conv3_2',input='pool2_2')    
    model.nodes['conv3_2'].W = model.nodes['conv3_1'].W
    model.nodes['conv3_2'].b = model.nodes['conv3_1'].b
    model.nodes['conv3_2'].params=[]
    model.add_node(BatchNormalization(),name='Bat3_1',input='conv3_1')
    model.add_node(BatchNormalization(),name='Bat3_2',input='conv3_2')
    conv4_1 = Convolution2D(384,3,3,activation='relu',border_mode='same') 
    conv4_2 = copy.deepcopy(conv4_1)
    model.add_node(conv4_1,name='conv4_1',input='Bat3_1')    
    model.add_node(conv4_2,name='conv4_2',input='Bat3_2')    
    model.nodes['conv4_2'].W = model.nodes['conv4_1'].W
    model.nodes['conv4_2'].b = model.nodes['conv4_1'].b
    model.nodes['conv4_2'].params=[]
    model.add_node(BatchNormalization(),name='Bat4_1',input='conv4_1')
    model.add_node(BatchNormalization(),name='Bat4_2',input='conv4_2')

    conv5_1 = Convolution2D(256,3,3,activation='relu',border_mode='same') 
    conv5_2 = copy.deepcopy(conv5_1)
    model.add_node(conv5_1,name='conv5_1',input='Bat4_1')    
    model.add_node(conv5_2,name='conv5_2',input='Bat4_2')    
    model.nodes['conv5_2'].W = model.nodes['conv5_1'].W
    model.nodes['conv5_2'].b = model.nodes['conv5_1'].b
    model.nodes['conv5_2'].params=[]
    model.add_node(BatchNormalization(),name='Bat5_1',input='conv5_1')
    model.add_node(BatchNormalization(),name='Bat5_2',input='conv5_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool5_1',input='Bat5_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool5_2',input='Bat5_2')

    model.add_node(Flatten(),name='Flat6_1',input='pool5_1')
    model.add_node(Flatten(),name='Flat6_2',input='pool5_2')


    dense7_1 = Dense(4096,activation='relu')
    dense7_2 = copy.deepcopy(dense7_1)
    model.add_node(dense7_1,name='dense7_1',input='Flat6_1')
    model.add_node(dense7_2,name='dense7_2',input='Flat6_2')
    model.nodes['dense7_2'].W = model.nodes['dense7_1'].W
    model.nodes['dense7_2'].b = model.nodes['dense7_1'].b
    model.nodes['dense7_2'].params =[]

    dense8 = Dense(4096,activation='relu')
    model.add_node(dense8,name='dense8',inputs=['dense7_1','dense7_2'],merge_mode='sum')
    dense9 = Dense(4096)
    model.add_node(dense9,name='dense9',input='dense8')
    model.add_node(Dropout(0.25), name = 'dropout10',input='dense9')
    model.add_node(Dense(13),name='dense11',input='dropout10')
    model.add_node(Activation('softmax'),name='softmax',input='dense11')
    model.add_output(name='out',input='softmax')

    return model


