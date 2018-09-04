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
	img = cv2.resize((cv2.cvtColor(img, cv2.COLOR_BGR2YUV)),(img_cols,img_rows))
	img = (img / 127.5) - 1.0
	return img[7:13,:]

index = 4475
X = []
y = []
	
#resized image dimension in training
img_rows = 16
img_cols = 32

# create adjusted steering measurements for the side camera images
steering_correction = 0.2  # this is a parameter to tune

# Load training data
df = pd.read_csv('driving_log.csv', header=None)		
print(df.head())

plt.imsave("center-final.png", preprocess_image(plt.imread(os.path.join('IMG', df[0][index].split('/')[-1]))))
plt.imsave("left-final.png", preprocess_image(plt.imread(os.path.join('IMG', df[1][index].split('/')[-1]))))
plt.imsave("right-final.png", preprocess_image(plt.imread(os.path.join('IMG', df[2][index].split('/')[-1]))))
			
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

plt.hist(y_train, bins=20)
plt.title("Steering Angle Distribution after adding flipped images")
plt.xlabel("Steering Angle")
plt.savefig("Flipped-Steering.png")
