
# %% [markdown]
# ## <span style="color:#873600">Importing libraries</span>

# %%
# run this code
import numpy as np
import random
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import imblearn   #synthesize more datapoints from minority classes

# %% [markdown]
# ## <span style="color:#873600">Task 1: Data loading</span>

# %% [markdown]
# In the **next** cell create methods to load the sensor data from numpy files.

# %%
#run this code

root = 'Project1/resources/gaze-detection_npy-data/' #You should change this to where you keep the data (use kaggle data)

#loads data from multiple files
def load_raw_data_class1():
    test_X1= np.load(f'{root}vid_1_gaze_test_X.npy')
    test_Y1= np.load(f'{root}vid_1_gaze_test_Y.npy')

    test_X2= np.load(f'{root}vid_2_gaze_test_X.npy')
    test_Y2= np.load(f'{root}vid_2_gaze_test_Y.npy')

    #You can load more videos if you would like here (optional extra credit)...
    
    test_X3= np.load(f'{root}vid_5_gaze_test_X.npy')
    test_Y3= np.load(f'{root}vid_5_gaze_test_Y.npy')
    
    test_X4= np.load(f'{root}vid_7_gaze_test_X.npy')
    test_Y4= np.load(f'{root}vid_7_gaze_test_Y.npy')

    return (test_X1, test_X2,test_X3, test_X4), (test_Y1, test_Y2, test_Y3, test_Y4)  #make sure to add them to your tuples too

def load_raw_data_class2():
    gaze_X1= np.load(f'{root}vid_1_gaze_GT_X.npy')
    gaze_Y1= np.load(f'{root}vid_1_gaze_GT_Y.npy')

    gaze_X2= np.load(f'{root}vid_2_gaze_GT_X.npy')
    gaze_Y2= np.load(f'{root}vid_2_gaze_GT_Y.npy')

    #You can load more videos if you would like here (optional extra credit)...
    
    gaze_X3= np.load(f'{root}vid_5_gaze_GT_X.npy')
    gaze_Y3= np.load(f'{root}vid_5_gaze_GT_Y.npy')

    gaze_X4= np.load(f'{root}vid_7_gaze_GT_X.npy')
    gaze_Y4= np.load(f'{root}vid_7_gaze_GT_Y.npy')

    return (gaze_X1, gaze_X2, gaze_X3, gaze_X4), (gaze_Y1, gaze_Y2, gaze_Y3, gaze_Y4) #make sure to add them to your tuples too

def load_data_class1():
    tuple_X, tuple_Y = load_raw_data_class1()
    test_points_X = np.concatenate(tuple_X, axis=1)
    test_points_Y = np.concatenate(tuple_Y, axis=1)

    return test_points_X, test_points_Y

def load_data_class2():
    tuple_X, tuple_Y = load_raw_data_class2()
    gaze_GT_X=np.concatenate(tuple_X, axis=1)
    gaze_GT_Y=np.concatenate(tuple_Y, axis=1)

    return gaze_GT_X, gaze_GT_Y

# %%
#printing out our data...

class1_X, class1_Y = load_data_class1()
class2_X, class2_Y = load_data_class2()

print(class1_X.shape, class1_Y.shape)
print(class2_X.shape, class2_Y.shape)


# %% [markdown]
# ## <span style="color:#873600">Task 2: choose wisely your features.</span>
# Add features that allow you to reliably identify classes.

# %%
# Featurize each subject's gaze trajectory in the file
# This method is run once for each video file for each condition
# X, Y, Z, W are each a loaded video
def featurize_input(X, Y):
    out = []
    # Where i is each subject in the file
    for i in range(len(X)):
        X_cord = X[i]
        Y_cord = Y[i] 
        fv = []
        
        # ADD CODE HERE #
        #add your features to fv (each feature here should be a single number)
        #you should replace the appends below with better features (mean, min, max, etc.)
        
        # Original Features #
        #fv.append(random.random())# use to test noise.. gives poor accuracy
        #fv.append(X_cord[0]) #initial X position
        #fv.append(X_cord[-1]) #final X position
        
        # My features #
        # X position 
        # Y position
        # Displacement from last point | note: first entry must have displacement 0
        # Reasoning: 
        #   X/Y Postions: class1 people may focus on different regions compared to non-autistic
        #   Displacement: class1 people may look around more/less often or may shift their gaze to further apart regions
        
        #calc average position
        avgx = 0
        for j in X_cord:
            avgx += j
            
        avgx = avgx/len(X_cord)
        
        avgy = 0
        for j in Y_cord:
            avgy += j
            
        avgy = avgy/len(Y_cord)
        
        
        #fv.append(avgx)
        fv.append(avgy)
        
        displacement = 0;
        for j in range(len(X_cord)):
            if j != 0:
                displacement += ((X_cord[j] - X_cord[j-1])**2) + ((Y_cord[j] - Y_cord[j-1])**2)** 0.5
                
        avgdisp = displacement/len(X_cord)
        fv.append(avgdisp)

        out.append(fv)

    out = np.array(out)

    return out

# %%
# run this code

def load_data_class1_fv():
    tuple_X, tuple_Y = load_raw_data_class1()

    fv_list = []
    #puts multiple loaded files into one continuous list
    for X, Y in list(zip(tuple_X, tuple_Y)):
      fv_list.append(featurize_input(X, Y))

    fv = np.concatenate(tuple(fv_list), axis=1)

    return fv

def load_data_class2_fv():
    tuple_X, tuple_Y = load_raw_data_class2()

    fv_list = []
    #puts multiple loaded files into one continuous list
    for X, Y in list(zip(tuple_X, tuple_Y)):
      fv_list.append(featurize_input(X, Y))

    fv = np.concatenate(tuple(fv_list), axis=1)

    return fv

# %%
# run this code

# get feature engineered vectors
class1_fv = load_data_class1_fv()
class2_fv = load_data_class2_fv()

# %% [markdown]
# ## <span style="color:#873600">Task 3: balance the datasets.</span>
# Balance your datasets. Explain your method and **comment** about your decision.

# %%
# ADD CODE HERE #

#synthesize 10 more class2 participants
# smote = imblearn.over_sampling.SMOTE(random_state=2022)

#print(control_fv)
print("Feature Vector Shapes")
print(class1_fv.shape, class2_fv.shape)

smote_X = np.concatenate((class1_fv,class2_fv),axis=0)

smote_Y = ["Class1"]*len(class1_fv)+["Class2"]*len(class2_fv)
print()
print("Shape of X and y before smote")
print(len(smote_Y)) 
print(smote_X.shape)

# print()
# print("Shape of X and y after smote")
# smote_out_X,smote_out_y = smote.fit_resample(smote_X, smote_Y)

# print(smote_out_X.shape,len(smote_out_y))

#Split the smote result back into our seperate vectors.
#class1 will be the same while class2 will have 10 more.

# class2_fv = smote_out_X[-35:]


# %% [markdown]
# Explanation: I balanced the data set by synthesizing more class2 participants to match the amount of class1 participants. I synthesized new points rather than remove class1 participants because there is already a relatively low sample of both. By keeping the sample size higher, the data will be  less influenced by outliers.

# %% [markdown]
# ## <span style="color:#873600">Task 4: dealing with ground truth.</span>
# Add the ground truth (labels) to the data.

# %%
# run this code

#Assigning groundtruth conditions to each participant. 
labels_class1 = [1.0] * len(class1_fv) 
labels_class2 = [0.0] * len(class2_fv)

#Make data and labels vectors (hint: np.concatenate)...
data = np.concatenate((class1_fv, class2_fv))
labels = np.concatenate((labels_class1, labels_class2))

###SANITY CHECK###
print(data.shape) # data (expected output: (60, #) ) -- Note: your y-dim may be different due to feature engineering or different number of videos
print(labels.shape) # labels (exptected output: (60,) )

#NOTE: If you chose to rebalance your data (task 3) then you may have more than 60 samples

# %% [markdown]
# ## <span style="color:#873600">Task 5: dealing with the spread of the features.</span>
# To know if we need to somehow normalize the data, it is useful to plot the spread of our features across the dataset. Write code to visualize the spread of our data (assuming that our data is contained in the variable 'X').
# 

# %%
# run this code 
print(data.shape)
for i in range(data.shape[1]):
    sns.kdeplot(data[:,i])


# %% [markdown]
# normalize the data. 

# %%
# run this code

scaler = RobustScaler()
data = scaler.fit_transform(data)
for i in range(data.shape[1]):
    sns.kdeplot(data[:,i])

# %% [markdown]
# ## <span style="color:#873600">Task 6: Examine the effect of feature engineering</span>
# %%
# run this code

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.30, random_state=5566)

#training the model
clf = RandomForestClassifier() #NOTE: this used to use SVC()
clf.fit(xtrain, ytrain)
cv_scores = cross_val_score(clf, xtrain, ytrain, cv=10)
print('Average Cross Validation Score from Training:', cv_scores.mean(), sep='\n', end='\n\n\n')

#testing the model
ypred = clf.predict(xtest)
cm = confusion_matrix(ytest, ypred)
cr = classification_report(ytest, ypred)

print('Confusion Matrix:', cm, sep='\n', end='\n\n\n')
print('Test Statistics:', cr, sep='\n', end='\n\n\n')
print('Testing Accuracy:', accuracy_score(ytest, ypred))


# %% [markdown]
# Explain your model performance: 
# 
# The data is split so that 70% of the data is used for training and remaining 30% percent, the test data, will be classified.
# 
# The average cross validation score shown is based on the average accuracy of the model splitting the training data into groups of ten, and for each data sample in the set, classifying it after training on the nine other samples. This gives an idea of the models performance in classifying data that it has already seen before
# 
# The testing accuracy score is the average accuracy when, for each test data sample, classify it using the training data. It has not trained on this test data, so understandably the accuracy is (usually) lower.
# 
# My features included the average x postion, average y position, and the average displacement between two x,y coordinate points. I will refer to the features as <b>x</b>, <b>y</b>, and <b>d</b>.
# 
# trial            | <b>x</b> | <b>y</b> | <b>d</b>| testing accuracy | cross validation score
# -----------------|----------|----------|---------|------------------|------------------------
#  1               | yes      | no       | no      | 90.5%            | 85.0%
#  2               | no       | yes      | no      | 95.2%            | 81.5%
#  3               | no       | no       | yes     | 95.2%            | 96.0%
#  4               | yes      | yes      | no      | 90.5%            | 90.0%
#  5               | yes      | no       | yes     | 90.5%            | 98.0%
#  6               | no       | yes      | yes     | 95.2%            | 98.0%
#  7               | yes      | yes      | yes     | 95.2%            | 96.0%
# 
# 
# My best feature combination seems to be from using <b>y</b> and <b>d</b>. Using all features resulted in a lower validation score with no significant change to the testing accuracy. Using <b>y</b> alone gives a high testing accuracy but relatively low cross validation score. Using <b>d</b> alone already gives a high testing accuracy and validation score, and adding the other features only seems to improve or have no effect on scores.
