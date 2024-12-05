# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:16:12 2024

@author: Poll
"""

# Question 1e
# ANN Model
import tensorflow as tf

#Model definition
My_NN_Model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(4,)),  
    tf.keras.layers.Dense(4, activation='sigmoid'),  
    tf.keras.layers.Dense(3, activation='relu'),     
    tf.keras.layers.Dense(3, activation='softmax')   
])

#Model summary
My_NN_Model.summary()

#Model compile 
My_NN_Model.compile(
    loss='categorical_crossentropy',  
    optimizer='adam',                
    metrics=['accuracy']          
)
#%%
# Question 2b
# ANN Model
My_NN_Model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(30, 30, 1)),  
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=2, strides=(1, 1), padding='same', activation='relu'),  
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=4, strides=(1, 1), padding='same', activation='relu'),  
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(3, activation='softmax')  
])

# Display model summary
My_NN_Model.summary()

# Compile the model
My_NN_Model.compile(
    loss='categorical_crossentropy',  
    optimizer='adam',                
    metrics=['accuracy']             
)
#%%
# Part 3 of the final exam.
# ANN, CNN and LSTM
# Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional 
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
#%%
#Import data and split the data into training and test sets.
df = pd.read_csv('C:/Users/Poll/Documents/Fall2024/NN_final/Final_News_DF_Labeled_ExamDataset.csv')
# Data inspection
print(df.head()) 
print("Column names:", df.columns)  
print("Shape of data:", df.shape) 
# Convert the labels politics, football and science into labels 0, 1, 2.
label_encoder = LabelEncoder()
df['LABEL'] = label_encoder.fit_transform(df['LABEL'])
y = df['LABEL'].values 
x = df.drop(columns=['LABEL']).values  
print("x shape=", x.shape)  
print("y shape=", y.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"x_train shape= {x_train.shape}, y_train= {y_train.shape}")
print(f"x_test shape = {x_test.shape}, y_test = {y_test.shape}")
#%%
# ANN Model
My_ANN_Model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(300,)),  
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(64, activation='relu'),   
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(3, activation='softmax')  ])

My_ANN_Model.summary()
My_ANN_Model.compile(
    loss='sparse_categorical_crossentropy',  
    optimizer='adam',                
    metrics=['accuracy']             
)

Hist_ANN = My_ANN_Model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test))

ANN_loss, ANN_accuracy = My_ANN_Model.evaluate(x_test, y_test, verbose=2)
ANNpredictions=My_ANN_Model.predict([x_test])
print(f"ANN Model Test Loss= {ANN_loss:.4f}")
print(f"ANN Model Test Accuracy= {ANN_accuracy:.4f}")

print("The predictions, x_test are \n", ANNpredictions)
print("The shape of the predictions, x_test is \n", ANNpredictions.shape) 
print("The single prediction vector for x_test[2] is \n", ANNpredictions[2]) 
print("The max - final prediction label for x_test[2] is\n", np.argmax(ANNpredictions[2])) 
#%%
plt.plot(Hist_ANN.history['accuracy'], label='accuracy')
plt.plot(Hist_ANN.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Accuracy vs Epochs',fontsize=20)
plt.legend(loc='lower right',fontsize=20)
#%%
plt.plot(Hist_ANN.history['loss'], label='loss')
plt.plot(Hist_ANN.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Loss vs Epochs',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
#%%
predicted_labels = np.squeeze(np.array(ANNpredictions.argmax(axis=1)))
#print(predicted_labels)
ANN_CM=confusion_matrix(predicted_labels, y_test)
print("The confusion matrix is \n", ANN_CM)   

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(ANN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: ANN') 
#%%
x_train_reshaped_CNN = x_train.reshape(-1, 300, 1, 1)
x_test_reshaped_CNN = x_test.reshape(-1, 300, 1, 1)
print(f"x_train_reshaped_CNN= {x_train_reshaped_CNN.shape}, x_test_reshaped_CNN= {x_test_reshaped_CNN.shape}")

CNN_Model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(300,1,1), kernel_size=(3,1), filters = 2, activation='relu'),
  
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Conv2D(filters=56, kernel_size=(3, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    
    tf.keras.layers.Flatten(), 
    ## https://www.tutorialspoint.com/keras/keras_dense_layer.htm
    tf.keras.layers.Dense(units=128, activation='relu'),#https://keras.io/api/layers/core_layers/dense/
    tf.keras.layers.Dense(units=3, activation='softmax')  
])
CNN_Model.summary()

CNN_Model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=["accuracy"],
                 optimizer='adam') 
                 
Hist_CNN=CNN_Model.fit(x_train_reshaped_CNN,y_train, epochs=4, validation_data=(x_test_reshaped_CNN, y_test))
#%%
CNN_loss, CNN_accuracy = CNN_Model.evaluate(x_test_reshaped_CNN, y_test, verbose = 2)
print("The test accuracy is \n", CNN_accuracy)
CNNpredictions=CNN_Model.predict([x_test_reshaped_CNN])
print(f"CNN Model Test Loss: {CNN_loss:.4f}")
print(f"CNN Model Test Accuracy: {CNN_accuracy:.4f}")


print("The predictions, x_test_reshaped_CNN are \n", CNNpredictions)
print("The shape of the predictions, x_test_reshaped_CNN is \n", CNNpredictions.shape) 
print("The single prediction vector for x_test_reshaped_CNN[2] is \n", CNNpredictions[2]) 
print("The max - final prediction label for x_test_reshaped_CNN[2] is\n", np.argmax(CNNpredictions[2])) 
#%%

plt.plot(Hist_CNN.history['accuracy'], label='accuracy')
plt.plot(Hist_CNN.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Accuracy vs Epochs',fontsize=20)
plt.legend(loc='lower right',fontsize=20)
#%%
plt.plot(Hist_CNN.history['loss'], label='loss')
plt.plot(Hist_CNN.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Loss vs Epochs',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
#%%
predicted_labels_CNN = np.squeeze(np.array(CNNpredictions.argmax(axis=1)))
#print(predicted_labels_CNN)
CNN_CM=confusion_matrix(predicted_labels_CNN, y_test)
print("The confusion matrix is \n", CNN_CM)   

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(CNN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: CNN') 
#%%
# LSTM model
x_train_reshaped_LSTM = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)  
x_test_reshaped_LSTM = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 
print(f"x_train_reshaped_LSTM= {x_train_reshaped_LSTM.shape}, x_test_reshaped_LSTM= {x_test_reshaped_LSTM.shape}")

y_train_one_hot_LSTM = to_categorical(y_train, num_classes=3)  
y_test_one_hot_LSTM = to_categorical(y_test, num_classes=3)

print(f"y_train_one_hot_LSTM= {y_train_one_hot_LSTM.shape}, y_test_one_hot_LSTM= {y_test_one_hot_LSTM.shape}")

LSTM_model = tf.keras.models.Sequential([
    Bidirectional(LSTM(50, return_sequences=False), input_shape=(x_train_reshaped_LSTM.shape[1], 1)),
    Dense(3, activation='softmax')  
])

LSTM_model.compile(
                 loss=keras.losses.BinaryCrossentropy(from_logits=False),
                 metrics=["accuracy"],
                 optimizer='adam'
                 )
    
LSTM_model.summary()

#%%
Hist_LSTM = LSTM_model.fit(x_train_reshaped_LSTM, y_train_one_hot_LSTM, epochs=4, validation_data=(x_test_reshaped_LSTM, y_test_one_hot_LSTM))

LSTM_loss, LSTM_accuracy = LSTM_model.evaluate(x_test_reshaped_LSTM, y_test_one_hot_LSTM, verbose=2)
LSTMpredictions=LSTM_model.predict([x_test_reshaped_LSTM])
print(f"LSTM Model Test Loss: {LSTM_loss:.4f}")
print(f"LSTM Model Test Accuracy: {LSTM_accuracy:.4f}")

print("The predictions, x_test_reshaped_LSTM are \n", LSTMpredictions)
print("The shape of the predictions, x_test_reshaped_LSTM is \n", LSTMpredictions.shape) 
print("The single prediction vector for x_test_reshaped_LSTM[2] is \n", LSTMpredictions[2]) 
print("The max - final prediction label for x_test_reshaped_LSTM[2] is\n", np.argmax(LSTMpredictions[2])) 
#%%
plt.plot(Hist_LSTM.history['accuracy'], label='accuracy')
plt.plot(Hist_LSTM.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Accuracy vs Epochs',fontsize=20)
plt.legend(loc='lower right',fontsize=20)
#%%
plt.plot(Hist_LSTM.history['loss'], label='loss')
plt.plot(Hist_LSTM.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Loss vs Epochs',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
#%%
predicted_labels_LSTM = np.squeeze(np.array(LSTMpredictions.argmax(axis=1)))
#print(predicted_labels_LSTM)
LSTM_CM=confusion_matrix(predicted_labels_LSTM, y_test)
print("The confusion matrix is \n", LSTM_CM)   

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(LSTM_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
ax.set_xlabel('True labels') 
ax.set_ylabel('Predicted labels')
ax.set_title('Confusion Matrix: LSTM') 