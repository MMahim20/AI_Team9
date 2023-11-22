from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import matplotlib.image as mpimg
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# adapted from the tutorial: https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
optimizer = tf.keras.optimizers.legacy.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Train network
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 30  # For better accuracy, increase the number of epochs
batch_size = 86

# Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, rotation_range=10, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=False, vertical_flip=False)

datagen.fit(X_train)

# Fit the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(X_val, y_val),
                    verbose=2, steps_per_epoch=X_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction])

############## VISUALIZATIONS
# Create visualizations showing prediction outcomes on training data

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].legend(loc='best', shadow=True)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_val, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plot_confusion_matrix(confusion_mtx, classes=range(10))

############## TESTING
# Use test data set to get final predicted goodness
test_df = pd.read_csv("test.csv")

# Normalize the data to be on [0..1] scale
X_test = test_df / 255.0

# Reshape the data to be an image in 3 dimensions 28pixels x 28pixels x 1 channel for the Keras CNN
X_test = X_test.values.reshape(-1,28,28,1)

# get test results
results = model.predict(X_test)

# get the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)