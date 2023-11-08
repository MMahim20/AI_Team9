from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

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
clf = RandomForestClassifier(max_depth=2, random_state=0)

# Train model with training data
###CHANGE
clf.fit(X_train, y_train)

# Use validation data to test accuracy during training
###CHANGE
y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Training Accuracy: ", accuracy)

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
y_pred = clf.predict(X_test)

# output submission file
output = pd.DataFrame({'id': test_df.id, 'target': y_pred})
output.to_csv('submission.csv', index=False)