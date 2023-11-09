from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import keras_nlp
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# adapted from the tutorial: https://www.kaggle.com/code/alexia/kerasnlp-starter-notebook-disaster-tweets



EPOCHS = 2
AUTO = tf.data.experimental.AUTOTUNE

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Import and explore data
df = pd.read_csv("Project2/train.csv")
print(df)
NUM_TRAINING_EXAMPLES = df.shape[0]
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
STEPS_PER_EPOCH = int(NUM_TRAINING_EXAMPLES)*TRAIN_SPLIT // BATCH_SIZE

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)

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
custom_optimizer = keras.optimizers.Adam(learning_rate=1e-5)
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), #'binary_crossentropy',
    optimizer=custom_optimizer,
    metrics= ["accuracy"]  
)

# Fit
history = classifier.fit(x=X_train,
                         y=y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS, 
                         validation_data=(X_val, y_val)
                        )

def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        display_labels=["Not Disaster","Disaster"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp+((fn+fp)/2))

    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))

# Use validation data to test accuracy during training
y_pred = classifier.predict(X_val)

classifier.save('tweet_disaster.model')



print("y_test shape:", y_val.shape)
print("y_pred shape:", y_pred.shape)
print("y_test values:", y_val)
print("y_pred values:", y_pred)

displayConfusionMatrix(y_train, y_pred, "Training")
# accuracy = accuracy_score(y_test, y_pred)
# print("Training Accuracy: ", accuracy)

# Fine-tune pre-trained model on tweets


############## VISUALIZATIONS
# Create visualizations showing prediction outcomes on training data


############## TESTING
# Use test data set to get final predicted goodness
test_df = pd.read_csv("Project2/test.csv")

# Get feature matrix from test data
X_test = test_df['text']
print(X_test)

###CHANGE
y_pred = classifier.predict(X_test)

# output submission file
### CHANGE based on y_pred
output = pd.DataFrame({'id': test_df.id, 'target': y_pred})
output.to_csv('submission.csv', index=False)
