import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Cropping2D, Conv2D, Lambda, Dropout, Flatten, Dense, ELU
from keras.utils import plot_model

def preprocess_image(img):
	return cv2.resize((cv2.cvtColor(img, cv2.COLOR_BGR2YUV)),(img_cols,img_rows))


X = []
y = []
	
#resized image dimension in training
img_rows = 16
img_cols = 32

# create adjusted steering measurements for the side camera images
steering_correction = 0.2  # this is a parameter to tune

# Load training data
df = pd.read_csv('driving_log.csv', header=None)						
for row in df.itertuples():
	img_center = preprocess_image(plt.imread(os.path.join('IMG', row[1].split('/')[-1])))
	img_left = preprocess_image(plt.imread(os.path.join('IMG', row[2].split('/')[-1])))
	img_right = preprocess_image(plt.imread(os.path.join('IMG', row[3].split('/')[-1])))
	steering_center = float(row[4])
	steering_left = steering_center + steering_correction
	steering_right = steering_center - steering_correction
	X.extend([img_center, img_left, img_right])
	y.extend([steering_center, steering_left, steering_right])	

X_train = np.array(X).astype('float32')
y_train = np.array(y).astype('float32')


# Augmentation - Add flipped images and labels
X_train = np.concatenate((X_train, [np.fliplr(x) for x in X_train]), axis=0)
y_train = np.concatenate((y_train, -y_train), axis=0)


# Split training and validation set
X_train, y_train = shuffle(X_train, y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)


# Use commaai.model with Cropping for training
model = Sequential()
model.add(Cropping2D(cropping=((7,3),(0,0)), input_shape=(img_rows,img_cols,3)))
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(6,img_cols,3)))
model.add(Conv2D(16, kernel_size=8, strides=(4, 4), padding="same", activation='elu'))
model.add(Conv2D(32, kernel_size=5, strides=(2, 2), padding="same", activation='elu'))
model.add(Conv2D(64, kernel_size=5, strides=(2, 2), padding="same", activation='elu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(ELU())
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(1))
model.summary()

model.compile(optimizer='Nadam', loss='mse')
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=2, save_best_only=True, monitor='val_loss')
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=2)


# Train the model and save the best weights
model.fit(X_train, y_train, epochs=20, batch_size=256, shuffle=True, verbose=1, validation_data=(X_val, y_val), callbacks=[checkpointer, callback])


# plot and save model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)