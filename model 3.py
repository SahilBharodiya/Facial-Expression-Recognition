#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random


#%%
data = pd.read_csv('facial_data.csv')


#%%
X = []
tmp_X = data.drop(columns=['ID'])
y = data['ID']

tmp_X = np.array(tmp_X)


#%%
for i in range(len(data)):
    img = tmp_X[i].reshape(48, 48, 1)
    X.append(img)


#%%
X = np.array(X)
y = np.array(y)


#%%
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


#%%
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


#%%
num_classes = 7

model = tf.keras.models.Sequential()
#1st convolution layer
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

#2nd convolution layer
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

#3rd convolution layer
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

model.add(tf.keras.layers.Flatten())

#fully connected neural networks
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(num_classes))

model.load_weights('facial_expression_model_weights.h5')

model.summary()


#%%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_X, train_y, epochs=10, validation_data=(test_X, test_y))


#%%
loss, acc = model.evaluate(test_X, test_y, verbose=2)


#%%
'''model.save('Facial_Expression_Recognition.h5')'''