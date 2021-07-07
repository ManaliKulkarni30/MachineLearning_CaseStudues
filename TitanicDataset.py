# ===================
# Imports
# ===================
import pandas as pd
import numpy as np
import seaborn as sb
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ===================
# ML Operation
# ===================
def TitanicLogistic():
    #print("Inside logistic function")

    #step 1 - Load Data
    titanic_data = pd.read_csv("TitanicDataset.csv")

    # Data Analysis
    print("First five records of dataset : ")
    print(titanic_data.head())
    """print("Total Number of records are : ",len(titanic_data))
    print(titanic_data.info())"""

    #Step 2 - Analyse the data
    """print("Visualization of survived and non survived passengers : ")
    figure()
    countplot(data=titanic_data,x="Survived").set_title("Survived v/s Non-Survived")
    show()

    print("Visualization according to gender : ")
    figure()
    countplot(data=titanic_data,x="Survived",hue="Sex").set_title("Visualization according to Sex")
    show()

    print("Visualization according to Passenger Class : ")
    figure()
    countplot(data=titanic_data,x="Survived",hue="Pclass").set_title("Visualization according to Passenger class")
    show()

    print("Survived v/s Unsurvived based on Age : ")
    figure()
    titanic_data["Age"].plot.hist().set_title("Visualization according to Age")
    show()"""

    #Step 3- Data Cleaning (Data Wrangling)
    titanic_data.drop("zero",axis=1,inplace=True)
    print("Data after column removal : ")
    print(titanic_data.head())

    Sex = pd.get_dummies(titanic_data["Sex"])
    print(Sex.head())
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print("Ssx column after updation : ")
    print(Sex.head())

    Pclass = pd.get_dummies(titanic_data["Pclass"])
    print(Pclass.head())
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print("Pclass column after updation : ")
    print(Pclass.head())

    #Concat Sex and P class field in our dataset
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
    print("Data after concatination : ")
    print(titanic_data.head())

    #Removing uneccesary fields
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head())

    #Divide the dataset into X and Y
    x = titanic_data.drop("Survived",axis=1)
    y = titanic_data["Survived"]

    #split the data for training and testing purpose
    x_train,x_test,y_train,y_test = train_test_split( x, y, test_size = 0.5, random_state = 0 )

    obj = LogisticRegression(max_iter=2000)

    #Step 4 - Train the dataset
    obj.fit(x_train,y_train)

    #Step 5 - Testing
    output = obj.predict(x_test)

    print("Accuracy of the given dataset is : ")
    print(accuracy_score(output,y_test)*100)

    print("Confusion Matrix is : ")
    print(confusion_matrix(y_test,output))




# =============
# Entry Point
# =============
def main():
    print("-------------Logistic Case Study--------------")

    TitanicLogistic()


# ==========
# Starter
# ==========
if __name__ == '__main__':
    main()
