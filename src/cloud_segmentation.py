import os
import skimage.io as io
import skimage.transform as trans

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers.normalization import BatchNormalization as bn

from keras.preprocessing.image import *

import glob
import cv2
import numpy as np
train_npy = []
myNames = glob.glob("/home/rutu/l8cloudmasks/x_train/*.tif")
for file in myNames:
    new = cv2.imread(file)
    new1 = cv2.resize(new,(256,256))
    
    
    train_npy.append(new1)
train_npy = np.array(train_npy)
train_npy = train_npy / 255

masks_npy = []
myNames = glob.glob("/home/rutu/l8cloudmasks/y_train/*.tif")
for file in myNames:
    new = cv2.imread(file)
    new1 = cv2.resize(new,(256,256))
    
    
    masks_npy.append(new1)
masks_npy = np.array(masks_npy)
masks_npy = masks_npy / 255

test = []
myNames = glob.glob("/home/rutu/l8cloudmasks/x_test/*.tif")
for file in myNames:
    new = cv2.imread(file)
    new1 = cv2.resize(new,(256,256))
    
    
    test.append(new1)
test = np.array(test)
test = test/255

input_shape = (256, 256, 3)
def UNet(input_shape,learn_rate=1e-3):
        l2_lambda = 0.0002
        DropP = 0.3
        kernel_size=3

        inputs = Input(input_shape)

        conv1 = Conv2D( 32, (kernel_size, kernel_size), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
        
        
        conv1 = bn()(conv1)
        
        conv1 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda)  )(conv1)

        conv1 = bn()(conv1)
        
        
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        pool1 = Dropout(DropP)(pool1)





        conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
        
        conv2 = bn()(conv2)

        conv2 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv2)

        conv2 = bn()(conv2)
        
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        pool2 = Dropout(DropP)(pool2)



        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)

        conv3 = bn()(conv3)
        
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv3)
        
        conv3 = bn()(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        pool3 = Dropout(DropP)(pool3)



        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
        conv4 = bn()(conv4)
        
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv4)
        
        conv4 = bn()(conv4)
        
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        pool4 = Dropout(DropP)(pool4)



        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
        
        conv5 = bn()(conv5)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv5)

        conv5 = bn()(conv5)
        
        up6 = concatenate([Conv2DTranspose(256,(2, 2), strides=(2, 2), padding='same')(conv5), conv4],name='up6', axis=3)

        up6 = Dropout(DropP)(up6)


        conv6 = Conv2D(256,(3, 3), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
        
        conv6 = bn()(conv6)

        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv6)

        conv6 = bn()(conv6)

        up7 = concatenate([Conv2DTranspose(128,(2, 2), strides=(2, 2), padding='same')(conv6), conv3],name='up7', axis=3)

        up7 = Dropout(DropP)(up7)

        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(up7)

        conv7 = bn()(conv7)
        
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv7)

        conv7 = bn()(conv7)

        up8 = concatenate([Conv2DTranspose(64,(2, 2), strides=(2, 2), padding='same')(conv7), conv2],name='up8', axis=3)

        up8 = Dropout(DropP)(up8)

        conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same', 
                       kernel_regularizer=regularizers.l2(l2_lambda) )(up8)

        conv8 = bn()(conv8)

        
        conv8 = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv8)

        conv8 = bn()(conv8)

        up9 = concatenate([Conv2DTranspose(32,(2, 2), strides=(2, 2), padding='same')(conv8), conv1],name='up9',axis=3)

        up9 = Dropout(DropP)(up9)

        conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
        
        conv9 = bn()(conv9)

        conv9 = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_lambda) )(conv9)
       
        conv9 = bn()(conv9)
       
        conv10 = Conv2D(3, (1, 1), activation='sigmoid', name='conv10')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

        #if(pretrained_weights):
         #   model.load_weights(pretrained_weights)

        return model

def train():

    print("loading data")
    #train_npy, test_npy, imgs_test = self.load_data()
    print("loading data done")
    model = UNet((256,256,3))
    print("got unet")

    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
    print('Fitting model...')
    model.fit(train_npy, masks_npy, batch_size=1, epochs=20, verbose=1,validation_split=0.2, shuffle=True)#, callbacks=[model_checkpoint])

    print('predict test data')
    imgs_mask_test = model.predict(test, batch_size=1, verbose=1)
    np.save('/home/rutu/l8cloudmasks/cloud_imgs_mask_test.npy', imgs_mask_test)
    #plt.scatter(range(20),result,c='r')


def save_img():

    print("array to image")
    imgs = np.load('/home/rutu/l8cloudmasks/cloud_imgs_mask_test.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("/home/rutu/l8cloudmasks/y_test/%d.tif"%(i))

train()
save_img()