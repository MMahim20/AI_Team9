from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_core as keras
import keras_nlp
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# adapted from the tutorial: https://www.kaggle.com/code/alexia/kerasnlp-starter-notebook-disaster-tweets

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Import and explore data
df = pd.read_csv("train.csv")
print(df)

# Create X and y matrices

# create the y label vector = result (Survived) column.
y = df['target']

# use the rest of the dataframe for matrix
X = df['text']

# explore the data
print('y:')
print(y)

print('\n--------------------')

print('\nX:')
print(X)

############## TRAINING AND VALIDATION TESTING
# Split training data into training and validation set

# 20% testing, 80% training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained embedding model
###CHANGE
preset= "distil_bert_base_en_uncased"

# Use a shorter sequence length.
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(preset,
                                                                   sequence_length=160,
                                                                   name="preprocessor_4_tweets"
                                                                  )

# Pretrained classifier.
classifier = keras_nlp.models.DistilBertClassifier.from_preset(preset,
                                                               preprocessor = preprocessor, 
                                                               num_classes=2)

classifier.summary()
# Train model with training data



# Use validation data to test accuracy during training




# Fine-tune pre-trained model on tweets


############## VISUALIZATIONS
# Create visualizations showing prediction outcomes on training data


############## TESTING
# Use test data set to get final predicted goodness
test_df = pd.read_csv("test.csv")

# Get feature matrix from test data
X_test = test_df['text']
print(X_test)

###CHANGE
# y_pred = 

# output submission file
### CHANGE based on y_pred
# output = pd.DataFrame({'id': test_df.id, 'target': y_pred})
# output.to_csv('submission.csv', index=False)