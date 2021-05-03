# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:09:53 2021

@author: Herr
"""

#Import Libraries
import numpy as np
import cv2
from  matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

#Set resolution for 
width=50
height=50
#Import dataset from folder with csv file
with open('English_characters/english.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    data=[]
    label=[]
    next(reader)
    for img,lab in reader:
        link='English_characters/'+img
        image_x=cv2.imread(link,cv2.IMREAD_GRAYSCALE)
        image_blur=cv2.GaussianBlur(image_x,(5,5),0)
        image_lap = np.uint8(cv2.Laplacian(image_blur,cv2.CV_64F))
        image_lap_blu = cv2.GaussianBlur(image_lap, (15, 15), 0)

        dim=(width,height)
        image_re=cv2.resize(image_lap_blu,dim, interpolation=cv2.INTER_AREA)
        data.append(np.array(image_re))
        label.append(str(lab))
print("Dataset acquired")   


#Encode labels as numeric        
le = LabelEncoder()
#Convert list type to np.array
label=np.array(label)
data=np.array(data)
Y_data = le.fit_transform(label)
X_data=data
#Split data to train/test
X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data, test_size = 0.25, random_state = 42)

#Reshape to increase dimension to add to 
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
Y_train=to_categorical(Y_train,62)
Y_test=to_categorical(Y_test,62)

dataGen= ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
dataGen.fit(X_train)
def Mymodel():
    NoFilter=32
    sizeFilter1=(5,5)
    sizeFilter2=(3,3)
    sizePool=(2,2)
    NoNode=500
        
    model= Sequential()
    model.add(Conv2D(NoFilter, sizeFilter1, activation='relu', input_shape=(width,height,1)))
    model.add(Conv2D(NoFilter, sizeFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add(Conv2D(NoFilter//2, sizeFilter2, activation='relu'))
    model.add(Conv2D(NoFilter//2, sizeFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    
    model.add(Dense(NoNode, activation='relu'))
    model.add(Dense(62, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
model=Mymodel()
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#Number of epochs and batch size to run model

sizeepochs = 200
batch_si = 55
history=model.fit(dataGen.flow(X_train,Y_train),
                    batch_size=batch_si,
                    epochs=sizeepochs,
                    callbacks=[EarlyStopping(monitor='loss', patience=15, min_delta=0.0001)],
                    shuffle=1)
Y_pred=model.predict(X_test)

#Confusion Matrix Calculation
cnf_matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
# Model Loss Plot
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show();

#Model Accuracy Plot
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show();

#HeatMap Visualization
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cnf_matrix, annot=True, fmt='d',xticklabels=le.classes_, yticklabels=le.classes_,cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

