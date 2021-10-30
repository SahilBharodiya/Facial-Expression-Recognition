#%%
import os  # For path loading
import cv2  # For working with images
import numpy as np  # Array related work
import matplotlib.pyplot as plt  # For ploting images
from sklearn.model_selection import train_test_split  # Deviding data in training, testing and validation
import tensorflow as tf
from tensorflow.keras import layers, models # For neural network
import random  # To assume numbers


# %%
A_e = 'image data/0'
Angry = os.listdir(A_e)

D_e = 'image data/1'
Disgust = os.listdir(D_e)

F_e = 'image data/2'
Fear = os.listdir(F_e)

H_e = 'image data/3'
Happy = os.listdir(H_e)

S_e = 'image data/4'
Sad = os.listdir(S_e)

Sp_e = 'image data/5'
Surprise = os.listdir(Sp_e)

N_e = 'image data/6'
Neutral = os.listdir(N_e)


# %%
X = []
y = []


# %%
for img in Angry:
    img_path = os.path.join(A_e, img)
    X.append(cv2.imread(img_path))
    y.append(0)
print("Done")

for img in Disgust:
    img_path = os.path.join(D_e, img)
    X.append(cv2.imread(img_path))
    y.append(1)
print("Done")

for img in Fear:
    img_path = os.path.join(F_e, img)
    X.append(cv2.imread(img_path))
    y.append(2)
print("Done")

for img in Happy:
    img_path = os.path.join(H_e, img)
    X.append(cv2.imread(img_path))
    y.append(3)
print("Done")

for img in Sad:
    img_path = os.path.join(S_e, img)
    X.append(cv2.imread(img_path))
    y.append(4)
print("Done")

for img in Surprise:
    img_path = os.path.join(Sp_e, img)
    X.append(cv2.imread(img_path))
    y.append(5)
print("Done")

for img in Neutral:
    img_path = os.path.join(N_e, img)
    X.append(cv2.imread(img_path))
    y.append(6)
print("Done")


# %%
X = np.array(X) / 255.0
y = np.array(y)


#%%
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)


#%%
num_classes = 7

model = tf.keras.models.Sequential()
#1st convolution layer
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 3)))
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

#model.load_weights('facial_expression_model_weights.h5')


#%%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


#%%
history = model.fit(train_X, train_y, epochs=5, validation_data=(test_X, test_y))


#%%
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_X,  test_y, verbose=2)
print("Accuracy = ", '%0.2f' % (test_acc*100), "%")