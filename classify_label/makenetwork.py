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

def vgg_label():
    model = Graph()
    model.add_input(name='input1',input_shape=(1,224,224))
    model.add_input(name='input2',input_shape=(1,224,224))
    
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

    conv8_1 = Convolution2D(512,3,3,activation='relu',border_mode='same') 
    conv8_2 = copy.deepcopy(conv8_1)
    model.add_node(conv8_1,name='conv8_1',input='pool7_1')    
    model.add_node(conv8_2,name='conv8_2',input='pool7_2')    
    model.nodes['conv8_2'].W = model.nodes['conv8_1'].W
    model.nodes['conv8_2'].b = model.nodes['conv8_1'].b
    model.nodes['conv8_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat6_1',input='conv6_1')
    #model.add_node(BatchNormalization(),name='Bat6_2',input='conv6_2')
    conv9_1 = Convolution2D(512,3,3,activation='relu',border_mode='same') 
    conv9_2 = copy.deepcopy(conv9_1)
    model.add_node(conv9_1,name='conv9_1',input='conv8_1')    
    model.add_node(conv9_2,name='conv9_2',input='conv8_2')    
    model.nodes['conv9_2'].W = model.nodes['conv9_1'].W
    model.nodes['conv9_2'].b = model.nodes['conv9_1'].b
    model.nodes['conv9_2'].params=[]
    conv10_1 = Convolution2D(512,3,3,activation='relu',border_mode='same') 
    conv10_2 = copy.deepcopy(conv10_1)
    model.add_node(conv10_1,name='conv10_1',input='conv9_1')    
    model.add_node(conv10_2,name='conv10_2',input='conv9_2')    
    model.nodes['conv10_2'].W = model.nodes['conv10_1'].W
    model.nodes['conv10_2'].b = model.nodes['conv10_1'].b
    model.nodes['conv10_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat7_1',input='conv7_1')
    #model.add_node(BatchNormalization(),name='Bat7_2',input='conv7_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool10_1',input='conv10_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool10_2',input='conv10_2')

    conv11_1 = Convolution2D(512,3,3,activation='relu',border_mode='same') 
    conv11_2 = copy.deepcopy(conv11_1)
    model.add_node(conv11_1,name='conv11_1',input='pool10_1')    
    model.add_node(conv11_2,name='conv11_2',input='pool10_2')    
    model.nodes['conv11_2'].W = model.nodes['conv11_1'].W
    model.nodes['conv11_2'].b = model.nodes['conv11_1'].b
    model.nodes['conv11_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat6_1',input='conv6_1')
    #model.add_node(BatchNormalization(),name='Bat6_2',input='conv6_2')
    conv12_1 = Convolution2D(512,3,3,activation='relu',border_mode='same') 
    conv12_2 = copy.deepcopy(conv12_1)
    model.add_node(conv12_1,name='conv12_1',input='conv11_1')    
    model.add_node(conv12_2,name='conv12_2',input='conv11_2')    
    model.nodes['conv12_2'].W = model.nodes['conv12_1'].W
    model.nodes['conv12_2'].b = model.nodes['conv12_1'].b
    model.nodes['conv12_2'].params=[]
    conv13_1 = Convolution2D(512,3,3,activation='relu',border_mode='same') 
    conv13_2 = copy.deepcopy(conv13_1)
    model.add_node(conv13_1,name='conv13_1',input='conv12_1')    
    model.add_node(conv13_2,name='conv13_2',input='conv12_2')    
    model.nodes['conv13_2'].W = model.nodes['conv13_1'].W
    model.nodes['conv13_2'].b = model.nodes['conv13_1'].b
    model.nodes['conv13_2'].params=[]
    #model.add_node(BatchNormalization(),name='Bat7_1',input='conv7_1')
    #model.add_node(BatchNormalization(),name='Bat7_2',input='conv7_2')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool13_1',input='conv13_1')
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool13_2',input='conv13_2')


    model.add_node(Flatten(),name='Flat13_1',input='pool13_1')
    model.add_node(Flatten(),name='Flat13_2',input='pool13_2')


    dense13_1 = Dense(4096,activation='relu')
    dense13_2 = copy.deepcopy(dense13_1)
    model.add_node(dense13_1,name='dense13_1',input='Flat13_1')
    model.add_node(dense13_2,name='dense13_2',input='Flat13_2')
    model.nodes['dense13_2'].W = model.nodes['dense13_1'].W
    model.nodes['dense13_2'].b = model.nodes['dense13_1'].b
    model.nodes['dense13_2'].params =[]

    dense14 = Dense(4096,activation='relu')
    model.add_node(dense14,name='dense14',inputs=['dense13_1','dense13_2'],merge_mode='sum')
    #model.add_node(Dropout(0.25), name = 'dropout10',input='dense9')
    #dense10 = Dense(1024,activation='relu')
    #model.add_node(dense10,name='dense11',input='dropout10')
    #model.add_node(Dropout(0.25),name='dropout12',input='dense11')
    model.add_node(Dense(13),name='dense15',input='dense14')
    model.add_node(Activation('softmax'),name='softmax',input='dense15')
    model.add_output(name='out',input='softmax')

    return model


