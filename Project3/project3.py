from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from keras.utils.np_utils import to_categorical
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

############## TRAINING AND VALIDATION TESTING
# Split training data into training and validation set

# 20% testing, 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############## SET UP THE NETWORK
# Create CNN network

# Train network

############## VISUALIZATIONS
# Create visualizations showing prediction outcomes on training data


############## TESTING
# Use test data set to get final predicted goodness
test_df = pd.read_csv("test.csv")
# X_test = test_df['text']
# print(X_test)