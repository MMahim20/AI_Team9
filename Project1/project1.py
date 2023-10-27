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
y = df['Survived']

# use the rest of the dataframe for matrix
X = df.drop(['Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)

print('y:')
print(y)

print('\n--------------------')
print('\nX:')
print(X)

# Split training data into training and validation set

# 20% testing, 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build random forest model
clf = RandomForestClassifier(max_depth=2, random_state=0)

# Build random forest model
clf = RandomForestClassifier(max_depth=2, random_state=0)

# Train model with training data
clf.fit(X_train, y_train)

# Use validation data to test accuracy during training
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Training Accuracy: ", accuracy)

# Use test data set to get final predicted goodness
test_df = pd.read_csv("test.csv")

# Get feature matrix from test data
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'Fare'], axis=1)
print(test_df)

y_pred = clf.predict(test_df)

# output submission file
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_pred})
output.to_csv('submission.csv', index=False)