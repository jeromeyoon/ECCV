from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, MaskedLayer, Flatten, RepeatVector, Reshape,Siamese
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import backend as K
#import theano.tensor as T
import pdb
"""
def softmaxdual(x):
    x1 = x
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

class Activations(MaskedLayer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1):
        super(Activations, self).__init__()
        # pdb.set_trace()
        self.activation = softmaxdual
        self.target = target
        self.beta = beta

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "activation": self.activation.__name__,
                "target": self.target,
                "beta": self.beta}
"""

# merge after dropout? or merge before dropout?
# merge_mode : 'sum' or 'concat'

    


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
    
    shared_dense5 = Dense(512)
    model.add_shared_node(shared_dense5,name ='dense5',inputs=['Flat4_1','Flat4_2'],outputs=['dense5_1','dense5_2'])
    model.add_node(Dense(512),name='concatlayer',inputs=['dense5_1','dense5_2'],merge_mode='sum')    
    #model.add_node(Dense(1024),name='dense6_1',input='dense5_1')
    #model.add_node(Dense(1024),name='dense6_2',input='dense5_2')
    #model.add_node(Dense(512),name='dense5_2',input='Flat4_2')
    #shared_dense6= Dense(1024)
    #model.add_node(shared_dense6,name='dense6',input='dense5',merge_mode='sum')
    #model.add_node(Dense(1024),name='dense6_1',input='dense5_1')
    #model.add_node(Dense(1024),name='dense6_2',input='dense5_2')
     
    model.add_node(Dense(1024),name='dense7',input='concatlayer')
    model.add_node(Dense(1024),name='dense8',input='dense7')
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
    
    



def eccvmodel_label():
	

    model = Graph()
    ### input 1 ####
    model.add_input(name='input1',input_shape=(1,42,42))
    model.add_input(name='input2', input_shape=(1,42,42))
    model.add_node(Convolution2D(64,9,9,activation='relu',border_mode='valid'),name='conv1-1',input='input1')    
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1-1',input='conv1-1')
    model.add_node(Convolution2D(128,7,7,activation='relu',border_mode='valid'),name='conv2-1',input='pool1-1')    
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2-1',input='conv2-1')
    model.add_node(Convolution2D(256,5,5,activation='relu',border_mode='valid'),name='conv3-1',input='pool2-1')    
    #### input 2 ####
    model.add_node(Convolution2D(64,9,9,activation='relu',border_mode='valid'),name='conv1-2',input='input2')    
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool1-2',input='conv1-2')
    model.add_node(Convolution2D(128,7,7,activation='relu',border_mode='valid'),name='conv2-2',input='pool1-2')    
    model.add_node(MaxPooling2D(pool_size=(2,2)),name='pool2-2',input='conv2-2')
    model.add_node(Convolution2D(256,5,5,activation='relu',border_mode='valid'),name='conv3-2',input='pool2-2')    
    ### Concate ####
    model.add_node(Convolution2D(512,1,1,border_mode='valid',activation='relu'), name='conv4', inputs=['conv3-1','conv3-2'], merge_mode='concat')
    model.add_node(Convolution2D(512,1,1,border_mode='valid',activation='relu'),name='conv5',input='conv4')
    
    model.add_node(Flatten(),name='Flat',input='conv5')
    model.add_node(Dense(512),name='dense1',input='Flat')
    #model.add_node(Dense(512,512),name='dense2',input='dense1')
    model.add_node(Dense(13),name='dense2',input='dense1')
    #model.add_node(Convolution2D(512,1,1,border_mode='valid',activation='relu'),name='conv5',input='conv4')
    #model.add_node(Convolution2D(13,1,1,border_mode='valid',activation='relu'),name='conv6',input='conv5')
    model.add_node(Activation('softmax'),name='softmax',input='dense2')
    model.add_output(name='out',input='softmax')


    return model




def makemergeconcatseq2048():
    
    model = Sequential()
    model.add(LSTM(8192, 4096, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(4096, 2048, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2048, 155))
    model.add(Activation('softmax'))
    return model

def makemergeconcat2():
    
    model = Sequential()
    model.add(LSTM(8192, 4096, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(4096, 512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(512, 67))
    model.add(Activation('softmax'))
    return model

def makemergesum2048():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(2048,2048, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='sum')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmmerge')
    model.add_node(Dense(2048,67),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

def makeobjvec_actnet():
    
    model = Graph()
    model.add_input(name='objvec',ndim=3)
    model.add_node(LSTM(2048,2048, return_sequences=True),name='inlstm',input='objvec')
    model.add_node(Dropout(0.2),name='dropin',input='inlstm')
    model.add_node(LSTM(2048,2048, return_sequences=False), name='lstmtrain', input='dropin')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmtrain')
    model.add_node(Dense(2048,67),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

def makemergeconcat2048obj():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(4096,2048, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='concat')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmmerge')
    model.add_node(Dense(2048,155),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

def makemergeconcat2048():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(4096,2048, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='concat')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmmerge')
    model.add_node(Dense(2048,155),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

def makemergeconcat1():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(4096,512, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='concat')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmmerge')
    model.add_node(Dense(512,67),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

# merge after dropout? or merge before dropout?
# merge_mode : 'sum' or 'concat'
def makemergedrop2():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(2048,512, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='sum')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmmerge')
    model.add_node(Dense(512,67),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

def makemerge2048drop2():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(2048,2048, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='sum')
    model.add_node(Dropout(0.2),name='droplstm',input='lstmmerge')
    model.add_node(Dense(2048,67),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model


def makemergenodrop():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(2048,512, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='sum')
    model.add_node(Dense(512,67),name='dense1',input='lstmmerge')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model

def makemergedrop5():
    
    model = Graph()
    model.add_input(name='rgb',ndim=3)
    model.add_input(name='of',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True),name='rgblstm',input='rgb')
    model.add_node(Dropout(0.2),name='droprgb',input='rgblstm')
    model.add_node(LSTM(4096,2048, return_sequences=True),name='oflstm',input='of')
    model.add_node(Dropout(0.2),name='dropof',input='oflstm')
    model.add_node(LSTM(2048,512, return_sequences=False), name='lstmmerge', inputs=['droprgb','dropof'], merge_mode='sum')
    model.add_node(Dropout(0.5),name='droplstm',input='lstmmerge')
    model.add_node(Dense(512,67),name='dense1',input='droplstm')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_output(name='out',input='softmax1')
    return model


def makegraph():
    model = Graph()
    model.add_input(name='input1',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True), name='lstm1', input='input1')
    model.add_node(Dropout(0.2),name='drop1',input='lstm1')
    model.add_node(LSTM(2048,512, return_sequences=True), name='lstm2', input='drop1')
    model.add_node(Dropout(0.2),name='drop2',input='lstm2')
    model.add_node(LSTM(512,512, return_sequences=False), name='lstm3', input='drop2')
    model.add_node(Dropout(0.2),name='drop3',input='lstm3')
    model.add_node(Dense(512,67),name='dense1',input='drop3')
    model.add_node(Dense(512,155),name='dense2',input='drop3')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_node(Activation('softmax'),name='softmax2',input='dense2')
    model.add_output(name='actout',input='softmax1')
    model.add_output(name='objout',input='softmax2')

    return model

def makeactnet():
    model = Sequential()
    model.add(LSTM(4096, 2048, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(2048, 512, return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(LSTM(1024, 512, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(512, 512, return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(512, 67))
    model.add(Activation('softmax'))
    return model

def makenetworkbasic():    
    model = Sequential()
    # model.add(Convolution2D(96, 3, 7, 7, border_mode='same')) 
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(96, 96, 7, 7))
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(256, 96, 5, 5, border_mode='same')) 
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(256, 256, 5, 5)) 
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(512, 256, 3, 3, border_mode='same')) 
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(512, 512, 3, 3)) 
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(2048, 512, 3, 3, border_mode='same')) 
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(2048, 2048, 3, 3)) 
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Flatten())
    # model.add(Dense(2048, 1024))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # 
    # model.add(Dense(1024, 101))
    # model.add(Activation('softmax'))
    
    # model.add(Dense(1024, 512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # 
    # model.add(Dense(1024, 256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    
    # # model.add(Embedding(512, 512))
    # model.add(RepeatVector(4096))
    # # the GRU below returns sequences of max_caption_len vectors of size 256 (our word embedding size)
    
    model.add(LSTM(4096, 2048, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(2048, 512, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(512, 155))
    model.add(Activation('softmax'))
    
    # model.add(LSTM(4096, 4096, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(4096, 2048, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(2048, 1024, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(1024, 512, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(512, 512, return_sequences=False))
    # # model.add(Dropout(0.2))
    # model.add(Dense(512, 67))
    # model.add(Activation('softmax'))

    return model
