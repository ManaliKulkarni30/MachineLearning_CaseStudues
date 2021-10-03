###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:03 October 2021
#About: predicting iris dataset using decision tree, random forest ,
#       K Nearest Neighbor(Practical Assignment)
#Roll No: 33
#Class : SY MSc CS
#
###########################################################################

#Required imports
import numpy as np
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


##########################################################################

#Helper Functions

def iris_decision_tree(x_train, x_test, y_train, y_test):
    obj = tree.DecisionTreeClassifier()

    obj.fit(x_train, y_train)

    result = obj.predict(x_test)

    print("Accuracy using Decision Tree: ",metrics.accuracy_score(y_test,result)*100)

def iris_KNN(x_train, x_test, y_train, y_test):
    obj = KNeighborsClassifier(n_neighbors=3)

    obj.fit(x_train, y_train)

    result = obj.predict(x_test)

    print("Accuracy using K Nearest Neighbor: ",metrics.accuracy_score(y_test,result)*100)

def iris_randomForest(x_train, x_test, y_train, y_test):
    obj = RandomForestClassifier(n_estimators=10)

    obj.fit(x_train, y_train)

    result = obj.predict(x_test)

    print("Accuracy using Random Forest: ",metrics.accuracy_score(y_test,result)*100)

##########################################################################

#Entry Point Function

def main():
    #Loading Dataset
    dataset = load_iris()

    print("Features of Dataset: ")
    print(dataset.feature_names)

    print("Target of Dataset: ")
    print(dataset.target_names)

    #Split the dataset into training dataset and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(dataset.data,dataset.target,test_size=0.3)
    #70% tarining and 30% testing data


    iris_decision_tree(x_train, x_test, y_train, y_test)
    iris_KNN(x_train, x_test, y_train, y_test)
    iris_randomForest(x_train, x_test, y_train, y_test)


##############################################################################

#Starter
if __name__ == '__main__':
    main()

##############################################################################

#Output:
"""D:\Python\CaseStudies>python Practical_Assignment.py
Features of Dataset:
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Target of Dataset:
['setosa' 'versicolor' 'virginica']
Accuracy using Decision Tree:  93.33333333333333
Accuracy using K Nearest Neighbor:  91.11111111111111
Accuracy using Random Forest:  91.11111111111111
"""
##############################################################################

#Definitions:
#Decision Tree:
"""
Decision tree is a type of supervised learning algorithm (having a pre-
defined target variable) that is mostly used in classification problems. It
works for both categorical and continuous input & output variables. In this
technique, we split the population (or sample) into two or more
homogeneous sets (or sub-populations) based on most significant splitter /
differentiator in input variables.
"""

#K Nearest Neighbor:
"""
Clustering is a type of Unsupervised learning.
This is very often used when you don’t have labeled data.
K-Means Clustering is one of the popular clustering algorithm.
The goal of this algorithm is to find groups(clusters) in the given data.
The K Means algorithm is iterative based, it repeatedly calculates the cluster centroids,
refining the values until they do not change much.
"""

#Random Forest:
"""
Random forests is a supervised learning algorithm. It can be used both for
classification and regression. It is also the most flexible and easy to use
algorithm.
A forest is comprised of trees. It is said that the more trees it has, the more
robust a forest is. Random forests creates decision trees on randomly selected
data samples, gets prediction from each tree and selects the best solution by
means of voting.
It also provides a pretty good indicator of the feature importance.
"""
