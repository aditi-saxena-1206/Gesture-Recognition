# CS386 : Term Project (Final Evaluation)
# Team Number 9
# Members: Uppala Sumana, Aditi Saxena, Manupati Vipulchandan
# Date of Submission : December 24, 2020
# File - Script to define and train the 3D CNN model for gesture prediction
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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import gc
import math
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

# subsetting csv files to the defined labels
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

for idx, row in tqdm(targets.iterrows(), total=len(targets)):
    if idx % 4 == 0:
        continue
    
    partition = []
    frames = make_thirty(TRAIN_SAMPLES_PATH + str(row['video_id']))
    if len(frames) == thirty:
        for frame in frames:
            frame = resize_frame(TRAIN_SAMPLES_PATH + str(row['video_id']) + '/' + frame)
            partition.append(rgb2gray(frame))
            if len(partition) == 15:
                if idx % 6 == 0:
                    new_frames_test.append(partition)
                    test_targets.append(row['label'])
                else:
                    new_frames.append(partition)
                    train_targets.append(row['label'])
                partition = []


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


#print(f"Training = {len(train_data)}/{len(train_targets)} samples/labels")
#print(f"Test = {len(test_data)}/{len(test_targets)} samples/labels")
#print(f"Validation = {len(cv_data)}/{len(cv_targets)} samples/labels")

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
    
        # 3D Convolution layers
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu',padding='same', name="conv1", data_format='channels_last')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),padding='same', data_format='channels_last')
        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3),padding='same', activation='relu', name="conv2", data_format='channels_last')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2),padding='same', data_format='channels_last')
        self.conv3 = tf.compat.v2.keras.layers.Conv3D(128, (3, 3, 3), activation='relu',padding='same', name="conv3", data_format='channels_last')
        self.pool3 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2),padding='same', data_format='channels_last')
   
        # LSTM & Flatten layers
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

# training the model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=32,
                    epochs=5)

# saving the weights to file
model.save_weights("model1.h5")
print("Saved model to disk")

model.save_weights('weights1/myweights', save_format='tf')

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

