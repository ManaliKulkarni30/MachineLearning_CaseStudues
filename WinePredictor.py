###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:15 June 2021
#About: Wine Predictor using K Nearest Neighbour Algorithm.
#
###########################################################################

#Required Python Package
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

###########################################################################

#Required Functions:

def WinePredictor():
    #Load datase
    #Here we are using the inbuilt dataset instead of using CSV
    wine = datasets.load_wine()

    #printing features names
    print(wine.feature_names)

    #printing target names
    print(wine.target_names)

    #printing top 5 records
    print(wine.data[0:5])

    #print the wine labels(0:class_0,1:class_1,2:class_2)
    print(wine.target)

    #Split the dataset into training dataset and testing dataset
    x_train, x_test, y_train, y_test = train_test_split(wine.data,wine.target,test_size=0.3)
    #70% tarining and 30% testing data

    #Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    #Train the Model using train set
    knn.fit(x_train,y_train)

    #Predict the response for the test data
    y_pred = knn.predict(x_test)

    #Model Accuracy, How Often the Model is Correct
    print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))

############################################################################

#Main function

def main():

    print("-----------------------------Manali Kulkarni-----------------------")

    print("------------Machine Learning Application---------------------")

    print("--------------Wine Predictor using K Nearest Neighbor Algorith----------------")

    WinePredictor()

############################################################################

#Starter

if __name__ == '__main__':
    main()
