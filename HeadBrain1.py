###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:11 April 2021
#About: Predicting size of brain using Linear Regression Algorithm(In inbuilt)
#
###########################################################################

#Required Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

##########################################################################

#Helper Functions
def MarvellousHeadBrain(path):
    dataset = pd.read_csv(path)
    print("Size of our datase is : ",dataset.shape)

    X = dataset["Head Size(cm^3)"].values
    Y = dataset["Brain Weight(grams)"].values
    X = X.reshape(-1, 1)
    obj = LinearRegression()

    obj.fit(X,Y)

    output = obj.predict(X)
    rsquare = obj.score(X,Y)
    print(rsquare)

##############################################################################

#Entry Point Function (Main Function)
def main():
    path = input("Enter name of dataset : ")
    MarvellousHeadBrain(path)

##############################################################################

#Starter
if __name__ == '__main__':
    main()
