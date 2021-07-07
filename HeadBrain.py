###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:10 April 2021
#About: Predicting size of brain using Linear Regression Algorithm(User defined)
#
###########################################################################

#Required Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###########################################################################

#Helper Functions
def meanData(arr):
    size = len(arr)
    sum = 0
    for i in range(size):
        sum = sum + arr[i]
    return sum

def MarvellousHeadBrain(path):
    dataset = pd.read_csv(path)
    print("Size of our datase is : ",dataset.shape)

    X = dataset["Head Size(cm^3)"].values
    Y = dataset["Brain Weight(grams)"].values

    print("Length of X : ",len(X))
    print("Length of Y : ",len(Y))

    Mean_X = meanData(X)
    Mean_Y = meanData(Y)

    n = 0
    d = 0

    for i in range(len(X)):
        n = n + ((X[i]-Mean_X)*(Y[i]-Mean_Y))
        d = d + ((X[i]-Mean_X)**2)

    m = n / d
    print("m : ",m)

    c = Mean_Y - (Mean_X*m)
    print("c : ",c)

    X_Start = np.min(X) - 200
    X_End = np.max(X) + 200

    x = np.linspace(X_Start,X_End)
    y = m*x + c

    plt.plot(x,y,color = 'r', label = "Line of Regression")
    plt.scatter(X,Y,color = 'b', label = "Data plot")

    plt.xlabel("Head size")
    plt.ylabel("Brain Weight")

    plt.legend()
    plt.show()

    Yp = []
    y = 0
    for i in range(len(X)):
         y = (m*X[i])+c
         Yp.append(y)

    #print("Yp : ",Yp)

    n = 0
    d = 0
    for i in range(len(X)):
        n = n + ((Yp[i]-Mean_Y)**2)
        d = d + ((Y[i] - Mean_Y)**2)

    r = n/d

    print("R Square is : ",r)

#############################################################################

#Entry Point Function
def main():
    path = input("Enter name of dataset : ")
    MarvellousHeadBrain(path)

##############################################################################

#Starter
if __name__ == '__main__':
    main()
