import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Lambda
from keras.optimizers import Adam
import os
import csv
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
data_dir_arr = ['/opt/carnd_p3/data/']

def get_samples():
    smpls = []
    for data_dir in data_dir_arr:
        with open(data_dir + 'driving_log.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for line in reader:
                line['center'] = data_dir + line['center'].strip()
                line['left'] = data_dir + line['left'].strip()
                line['right'] = data_dir + line['right'].strip()
                smpls.append(line)
    return smpls

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:      
                images.append(plt.imread(batch_sample['center']))
                angles.append(float(batch_sample['steering']))
                
                images.append(np.fliplr(plt.imread(batch_sample['center'])))
                angles.append(-1*float(batch_sample['steering']))
                
                images.append(plt.imread(batch_sample['left']))
                angles.append(float(batch_sample['steering'])+0.2)
                images.append(plt.imread(batch_sample['right']))
                angles.append(float(batch_sample['steering'])-0.2)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = get_samples()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
            
def create_model():
    mdl = Sequential()
    mdl.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
    mdl.add(Lambda(lambda x: (x-128.0)/128.0))
    mdl.add(Conv2D(24,5,strides=(2, 2),activation='elu',padding='valid'))
    mdl.add(Dropout(0.5))
    mdl.add(Conv2D(36,5,strides=(2, 2),activation='elu',padding='valid'))
    mdl.add(Dropout(0.5))
    mdl.add(Conv2D(48,5,strides=(2, 2),activation='elu',padding='valid'))
    mdl.add(Dropout(0.5))
    mdl.add(Conv2D(64,3,strides=(1, 1),activation='elu',padding='valid'))
    mdl.add(Dropout(0.5))
    mdl.add(Conv2D(64,3,strides=(1, 1),activation='elu',padding='valid'))
    mdl.add(Dropout(0.5))
    mdl.add(Flatten())
    mdl.add(Dense(100,activation='elu'))
    mdl.add(Dropout(0.3))
    mdl.add(Dense(50,activation='elu'))
    mdl.add(Dropout(0.2))
    mdl.add(Dense(10,activation='elu'))
    mdl.add(Dropout(0.1))
    mdl.add(Dense(1))
    optmzr = Adam(lr=LEARNING_RATE) 
    mdl.compile(optimizer=optmzr, loss='mean_squared_error', metrics=['accuracy'])
    return mdl

    
model = create_model()
model.summary()

checkpoint = ModelCheckpoint(filepath='./model.h5', monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=10)

hist = model.fit_generator(train_generator, 
            steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(len(validation_samples)/BATCH_SIZE), 
            epochs=EPOCHS, verbose=1,callbacks=[checkpoint,stopper])
#model.save('model.h5')
### plot the training and validation loss for each epoch
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
fig.tight_layout()
fig.savefig("loss.png")