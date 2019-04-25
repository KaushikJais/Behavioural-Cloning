import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
# import matplotlib.pyplot as plt
import csv


import keras
from keras.models  import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D,Lambda
from keras.optimizers import Adam
from keras.layers.convolutional import Cropping2D


images = []
measurements = []

#Parsing the CSV file provided by Udacity
lines =[]
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
for line in lines:
    if line[3]=='steering':
        continue
    for i in range(3): #To add images from all the cameras
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        measurement = float(line[3])
        measurements.append(measurement)
#         if i==0:
#             measurements.append(measurement)
#         elif i==1:
#             measurements.append(measurement+0.4)
#         elif i==2:
#             measurements.append(measurement-0.4)
#         images.append(cv2.flip(image_rgb,1))
#         if i==0:
#             measurements.append(measurement*-1.0)
#         elif i==1:
#             measurements.append((measurement+0.4)*-1.0)
#         elif i==2:
#             measurements.append((measurement-0.4)*-1.0)


X_train = np.array(images)
y_train = np.array(measurements)


#Modified NVIDIA Model for End-to-End Deep Learning for Self Driving Car
#Model architecture
model = Sequential()

#Data Pre-processing (Normalization,Cropping)
model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))

model.add(Conv2D(filters=24,kernel_size = (5,5),strides = (2,2),activation= 'relu'))
model.add(Conv2D(filters = 36,kernel_size = (5,5),strides = (2,2),activation = 'relu'))
model.add(Conv2D(filters = 48,kernel_size = (5,5),strides = (2,2),activation= 'relu'))

model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = (1,1),activation = 'relu'))
model.add(Conv2D(filters = 64,kernel_size = (3,3),strides = (1,1),activation = 'relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

#Compile and test
model.compile(loss='mse',optimizer = 'adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle = True,nb_epoch= 5,verbose=1)

#Save the Model
model.save('model_new1.h5')

    
