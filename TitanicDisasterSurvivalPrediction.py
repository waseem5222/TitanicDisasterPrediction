# linear algebra
import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
import math

def calculateGaussianProbability(x, mean, stdev):
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo

def calculate_likelihood_categorical(X_train, feat_name, feat_val, Y, label):
    X_train = X_train[X_train[Y]==label]
    p_x_given_y = len(X_train[X_train[feat_name]==feat_val]) / len(X_train)
    return p_x_given_y

def calculate_prior(X_train, Y_train):
    classes = sorted(list(X_train[Y_train].unique()))
    prior = []
    for i in classes:
        prior.append(len(X_train[X_train[Y_train]==i])/len(X_train))
    return prior

def getPredictions(X_train, X_test, Y_train):
    prior = calculate_prior(X_train, Y_train)
    #X_train = X_train.drop(["Survived"], axis=1)
    features = list(X_train.columns)[:-1]
    Y_pred = []
    for x in X_test:
        # calculate likelihood
        labels = sorted(list(X_train[Y_train].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):                
                likelihood[j] *=calculate_likelihood_categorical(X_train, features[i], x[i], Y_train, labels[j])
                 

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))
    return np.array(Y_pred)

test_data = pd.read_csv("input/test.csv")
train_data = pd.read_csv("input/train.csv")


print(train_data.head())
train_data =train_data.drop(["Ticket", "Cabin", "PassengerId", "Name"], axis=1)

print(train_data.info())
train_data.dtypes

print("\n Embarked options: ", train_data["Embarked"].value_counts())

print(train_data.describe())
print(train_data.isna())
print(train_data.isna().sum())




data = [train_data, test_data]

for dataset in data:
    mean = train_data["Age"].mean()
    std = train_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)

common_value = 'S'
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

genders = {"male": 0, "female": 1}
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

ports = {"S": 0, "C": 1, "Q": 2}

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

train_data["Age"] = pd.cut(train_data["Age"].values, bins=[0, 3, 17, 50, 80], labels=["Baby", "Child", "Adult", "Elderly"])
train_data["Fare"] = pd.cut(train_data["Fare"].values, bins=3, labels=[0,1,2])



Y_train = "Survived"
test_data  = test_data.drop(["Ticket", "Cabin", "PassengerId", "Name"], axis=1) 
test_data = test_data.iloc[:,:-1].values

#train, valid = train_test_split(train_data, test_size=0.2)
#X_test = valid.iloc[:,:-1].values

predictions = getPredictions(train_data, test_data, Y_train)

print(predictions)