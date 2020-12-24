# CS386 : Term Project (Final Evaluation)
# Team Number 9
# Members: Uppala Sumana, Aditi Saxena, Manupati Vipulchandan
# Date of Submission : December 24, 2020
# File - Script to define and train the 3D CNN model for gesture prediction - with data augmentation
#============================================
# importing required libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tqdm import tqdm

import seaborn as sbn
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import img_data_aug
import os
import gc
import math
import os
import random
from scipy import ndarray

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
import matplotlib

#=============================================
# functions to perform data augmentation

def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

#============================================
# defining class labels

LABELS = {
    "Swiping Right": 0,
    "Swiping Left": 1,
    "No gesture": 2,
    "Thumb Up": 3,
}

#=============================================
# defining path to dataset and csv files

TRAIN_SAMPLES_PATH = '/Users/supriyauppala/Downloads/videos/'
VAL_SAMPLES_PATH = '/Users/supriyauppala/Downloads/videos/'

targets = pd.read_csv('/Users/supriyauppala/Downloads/jester_train2.csv')
targets = targets[targets['label'].isin(LABELS.keys())]
targets['label'] = targets['label'].map(LABELS)
targets = targets[['video_id', 'label']]
targets = targets.reset_index()

targets_val = pd.read_csv('/Users/supriyauppala/Downloads/jester_validation2.csv')
targets_val = targets_val[targets_val['label'].isin(LABELS.keys())]
targets_val['label'] = targets_val['label'].map(LABELS)
targets_val = targets_val[['video_id', 'label']]
targets_val = targets_val.reset_index()

#=============================================
# functions to preprocess the images

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def resize_frame(frame):
    frame = img.imread(frame)
    frame = cv2.resize(frame, (64, 64))
    return frame

thirty = 30
def make_thirty(path):
    frames = os.listdir(path)
    frames_count = len(frames)
    if thirty > frames_count:
        frames += [frames[-1]] * (thirty - frames_count)
    elif thirty < frames_count:
        frames = frames[0:thirty]
    return frames

#============================================
# preprocess and split the training dataset

train_targets = []
test_targets = []

new_frames = []
new_frames_test = []

transdict = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
    }

for idx, row in tqdm(targets.iterrows(), total=len(targets)):
    if idx % 4 == 0:
        continue
    
    partition = []
    frames = make_thirty(TRAIN_SAMPLES_PATH + str(row['video_id']))
    folder_path=(TRAIN_SAMPLES_PATH + str(row['video_id']))
    image_lst = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (os.path.isfile(os.path.join(folder_path, f))and f != '.DS_Store')]
    print(image_lst)
    partition1 = []
    if len(frames) == thirty:
        for frame in frames:
            image_to_transform = sk.io.imread(TRAIN_SAMPLES_PATH + str(row['video_id'])+'/'+frame,plugin='matplotlib')
            frame = resize_frame(TRAIN_SAMPLES_PATH + str(row['video_id']) + '/' + frame)
            partition.append(rgb2gray(frame))
            transformed_image = None
            key = random.choice(list(transdict))
            transformed_image = transdict[key](image_to_transform)
            transformed_image = cv2.resize(transformed_image, (64, 64))
            print("Frame : ")
            print(frame)
            print(frame.shape)
            print("Transformed image : ")
            print(transformed_image)
            print(transformed_image.shape)
            partition1.append(rgb2gray(transformed_image))
            if len(partition) == 15:
                if idx % 6 == 0:
                    new_frames_test.append(partition)
                    test_targets.append(row['label'])
                else:
                    new_frames.append(partition)
                    train_targets.append(row['label'])
                partition = []
            if len(partition1) == 15:
                if idx % 6 == 0:
                    new_frames_test.append(partition1)
                    test_targets.append(row['label'])
                else:
                    new_frames.append(partition1)
                    train_targets.append(row['label'])
                partition1 = []

train_data = np.asarray(new_frames, dtype=np.float16)
del new_frames[:]
del new_frames

test_data = np.asarray(new_frames_test, dtype=np.float16)
del new_frames_test[:]
del new_frames_test

gc.collect()

#============================================
# preprocess and split the validation dataset


cv_targets = []
new_frames_cv = []
for idx, row in tqdm(targets_val.iterrows(), total=len(targets_val)):
    if idx % 4 == 0:
        continue

    partition = []
    # Frames in each folder
    frames = make_thirty(VAL_SAMPLES_PATH+str(row["video_id"]))
    for frame in frames:
        frame = resize_frame(VAL_SAMPLES_PATH+str(row["video_id"])+'/'+frame)
        partition.append(rgb2gray(frame))
        if len(partition) == 15:
            new_frames_cv.append(partition)
            cv_targets.append(row['label'])
            partition = []
                
cv_data = np.array(new_frames_cv, dtype=np.float16)
del new_frames_cv[:]
del new_frames_cv
gc.collect()


#============================================

# scaling the images - training data
scaler = StandardScaler(copy=False)
scaled_images  = scaler.fit_transform(train_data.reshape(-1, 15*64*64))
del train_data

scaled_images  = scaled_images.reshape(-1, 15, 64, 64, 1)

# scaling the images - test data
scaler = StandardScaler(copy=False)
scaled_images_test = scaler.fit_transform(test_data.reshape(-1, 15*64*64))
del test_data

scaled_images_test = scaled_images_test.reshape(-1, 15, 64, 64, 1)

#scaling the images - validation data
scaler = StandardScaler(copy=False)
scaled_images_cv  = scaler.fit_transform(cv_data.reshape(-1, 15*64*64))
del cv_data

scaled_images_cv  = scaled_images_cv.reshape(-1, 15, 64, 64, 1)
del scaler

#=============================================

# storing the final data after processing

y_train = np.array(train_targets, dtype=np.int8)
y_test = np.array(test_targets, dtype=np.int8)
y_val = np.array(cv_targets, dtype=np.int8)
del train_targets
del test_targets
del cv_targets



x_train = scaled_images
x_test = scaled_images_test
x_val = scaled_images_cv
del scaled_images
del scaled_images_test
del scaled_images_cv


gc.collect()

#==============================================
# the 3D Convolution Model

class Conv3DModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
    
        # Convolutions
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu',padding='same', name="conv1", data_format='channels_last')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),padding='same', data_format='channels_last')
        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu',padding='same', name="conv2", data_format='channels_last')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2),padding='same', data_format='channels_last')
        self.conv3 = tf.compat.v2.keras.layers.Conv3D(128, (3, 3, 3), activation='relu',padding='same', name="conv3", data_format='channels_last')
        self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), padding='same',data_format='channels_last')
   
        # LSTM & Flatten
        self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten =  tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(4, activation='softmax', name="output")

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
              
# training the model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=32,
                    epochs=5)

# saving the weights to file
model.save_weights("model1.h5")
print("Saved model to disk")

model.save_weights('weights2/myweights', save_format='tf')


#=================================================
# predicting for test data
np.unique(y_test, return_counts=True)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)
labels = list(LABELS.keys())

# creating a confusion matrix of the prediction result
cm = confusion_matrix(y_test, y_pred, normalize='true')
df_cm = pd.DataFrame(cm, range(4), range(4))

#plotting the confusion matrix as a heatmap
plt.figure(figsize=(10,7))
sbn.set(font_scale=1.4) # for label size
sbn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, xticklabels=labels, yticklabels=labels)
plt.show()

#================================================
# calculating accuracy

print(accuracy_score(y_test, y_pred))


print(precision_score(y_test, y_pred, average='macro'))


print(recall_score(y_test, y_pred, average='macro'))


print(f1_score(y_test, y_pred, average='macro'))

# end

