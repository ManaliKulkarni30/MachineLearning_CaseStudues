###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:10 April 2021
#About: Demo program for calculations in liner Regression
#
###########################################################################

#Required imports
import numpy as np

###########################################################################

#Helper Function
def Calculation():
    #Sample Data
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    #Mean of Data
    X_Mean = np.mean(X)
    Y_Mean = np.mean(Y)

    n = 0
    d = 0

    for i in range(len(X)):
        n = n + ((X[i]-X_Mean)*(Y[i]-Y_Mean))
        d = d + ((X[i]-X_Mean)**2)

    m = n / d

    print("X : ",X)
    print("Y : ",Y)

    c = Y_Mean - (X_Mean*m)

    print("m : ",m)
    print("c : ",c)

    Yp = []
    y = 0
    for i in range(len(X)):
         y = (m*X[i])+c
         Yp.append(y)

    print("Yp : ",Yp)

    n = 0
    d = 0
    for i in range(len(X)):
        n = n + ((Yp[i]-Y_Mean)**2)
        d = d + ((Y[i] - Y_Mean)**2)

    r = n/d

    print("R Square is : ",r)

#############################################################################

#Entry Point Function
def main():
    print("------------------Linear Regression--------------------")
    Calculation()

#############################################################################

#Starter
if __name__ == '__main__':
    main()
