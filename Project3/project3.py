from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Import and explore training data
df = pd.read_csv("train.csv")
print(df)

# Create X and y matrices
# create the y label vector
y = df['label']

# use the rest of the dataframe for matrix
X = df.drop(['label'], axis=1)

# Normalize the data to be on [0..1] scale
X = X / 255.0

# Reshape the data to be an image in 3 dimensions 28pixels x 28pixels x 1 channel for the Keras CNN
X = X.values.reshape(-1,28,28,1)

# encode y label vector to be one hot vectors. There are 10 digits (0-9).
y = to_categorical(y, num_classes=10)

############## TRAINING AND VALIDATION TESTING
# Split training data into training and validation set
# 20% testing, 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# show example
example = plt.imshow(X_train[0][:,:,0])
plt.show()

############## SET UP THE NETWORK
# Create CNN network


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Train network

############## VISUALIZATIONS
# Create visualizations showing prediction outcomes on training data


############## TESTING
# Use test data set to get final predicted goodness
test_df = pd.read_csv("test.csv")

# Normalize the data to be on [0..1] scale
X_test = test_df / 255.0

# Reshape the data to be an image in 3 dimensions 28pixels x 28pixels x 1 channel for the Keras CNN
X_test = X_test.values.reshape(-1,28,28,1)

# X_test = test_df['text']
# print(X_test)