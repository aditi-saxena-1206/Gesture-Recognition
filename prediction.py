# CS386 : Term Project (Final Evaluation)
# Team Number 9
# Members: Uppala Sumana, Aditi Saxena, Manupati Vipulchandan
# Date of Submission : December 24, 2020
# File - Script to real time capture and predict gestures
#============================================
# importing required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.preprocessing import StandardScaler

#============================================
# the 3D Convolution Model to load the weights
class Conv3DModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
    
        # 3D Convolution layers
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu',padding='same', name="conv1", data_format='channels_last')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),padding='same', data_format='channels_last')
        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3),padding='same', activation='relu', name="conv2", data_format='channels_last')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2),padding='same', data_format='channels_last')
        self.conv3 = tf.compat.v2.keras.layers.Conv3D(128, (3, 3, 3), activation='relu',padding='same', name="conv3", data_format='channels_last')
        self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2),padding='same', data_format='channels_last')
   
        # LSTM & Flatten
        self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten =  tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(4, activation='softmax', name="output")

    # implementing layers in-order
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)


# creating an instance of the model
model = Conv3DModel()

# defining the configurations of the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])
              
# loading the weights from the previously trained model
model.load_weights("weights1/myweights")

#===============================================
# function to scale the images uniformaly
def normaliz_data(np_data):
    scaler = StandardScaler()
    scaled_images  = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images
    
#===============================================
# execution begins here

# defining class labels    
classes = [
    "Swiping Right",
    "Sliding Left",
    "No gesture",
    "Thumb Up"
    ]

# staring video capture
to_predict = []
num_frames = 0
cap = cv2.VideoCapture(0)
classe =''

while(True):
    # capture frame-by-frame
    ret, frame = cap.read()
    
    # preprocessing on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(gray, (64, 64)))
    
    # after we have captured required number of frames
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = normaliz_data(frame_to_predict)
        predict = model.predict(frame_to_predict)
        classe = classes[np.argmax(predict)]
        
        # logging the output
        print('Classe = ',classe, 'Precision = ', np.amax(predict)*100,'%')
        to_predict = []
        
    # display the text on screen
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition',frame)
    
    # exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()







