# Gesture-Recognition

In this project we are a model to predict Hand Gestures in real-time.

# About the dataset
The 20BN - Jester Dataset (https://20bn.com/datasets/jester)   
Collection of labeled video clips that show humans performing pre-defined hand gestures in front of a laptop camera or webcam.  
Total number of videos   - 148,092  
Training Set              -  118,562  
Validation Set           - 14,787  
Test Set                  -   14,743  
Labels                   - 27  

# About the algorithm
1 - Data Preprocessing - images are greyscaled, resized and normalized  
2 - Taking fixed number of frames per video  
3 - Split into train and test data  
4 - The 3D CNN model for prediction can be seen in the figure below  
![alt text](https://github.com/aditi-saxena-1206/Gesture-Recognition/blob/main/model.png?raw=true)

The model was trained on a subset of the entire dataset, classifying the following classes
1) Swiping Right   
2) Swiping Left  
3) Thumb Up  
4) No Gesture  

The script for training the model is given in 'train_cnn.py'

The weights after training are stored in the folder 'weights1'

# Accuracy
The model, on test data, had the following accuracy scores
![alt text](https://github.com/aditi-saxena-1206/Gesture-Recognition/blob/main/train_cnn-accuracy.png?raw=true)

# Real-time Prediction
To use the mode, the script 'prediction.py' can be run using the command  
$ python3 prediction.py


# Improvement
We tried to improve the accuracy of the code using Data Augmentation to get a more varied dataset. The file 'shallow_dataaug.py' contains the script to tain the model with data augmentation.
