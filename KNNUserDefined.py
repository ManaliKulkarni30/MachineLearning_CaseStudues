###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:4th April 2021
#About: User defined version Of K Nearest Neighbor Algorithm
#
###########################################################################

#Required imports
from MyModules import *

###########################################################################

#Helper Functions(Naked Functions)
#1. Returns Euclidien Distance
def caculateDistance(x,y):
    return distance.euclidean(x,y)

##########################################################################

#2. Calculate Accuracy of Testing Model
def calcAccuracy(target_test,prediction):
    icnt = 0
    for i in range(len(target_test)):
        if target_test[i] != prediction[i]:
            icnt += 1

    accuracy = ((len(target_test)-icnt)/len(target_test))

    return accuracy

###########################################################################

#Class for K Nearest Neighbor
class Marvellous:
    def fit(self,training_data,training_target):
        #print("Data Training done")
        self.training_data = training_data
        self.training_target = training_target

    def predict(self,test_data):
        predictions = []
        for row in test_data:
            label = self.Shortest(row)
            predictions.append(label)
        return predictions

    def Shortest(self,row):
        minIndex = 0
        minDist = caculateDistance(row,self.training_data[0])
        for i in range(1,len(self.training_data)):
            Distance = caculateDistance(row,self.training_data[i])
            if Distance < minDist:
                minDist = Distance
                minIndex = i
        return self.training_target[minIndex]

############################################################################

#Helper Function
def MarvellousKNN():
    #print("Inside user defined KNN inmplementation")
    Line = "*"*50
    iris = load_iris()

    data = iris.data
    target = iris.target

    print("Actual Data:")
    print(Line)
    for i in range(len(iris)):
        print("ID: %d Label :%s Features:%s "%(i,iris.data[i],iris.target[i]))

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size = 0.5)

    print(Line)
    print("Training Data:")
    print(Line)
    for i in range(len(data_train)):
        print("ID: %d Label :%s Features:%s "%(i,data_train[i],target_train[i]))

    print(Line)
    print("Testing Data:")
    print(Line)
    for i in range(len(data_test)):
        print("ID: %d Label :%s Features:%s "%(i,data_test[i],target_test[i]))
    print(Line)
    #print("Data loaded successfully")

    mobj = Marvellous()

    mobj.fit(data_train,target_train)

    ret = mobj.predict(data_test)

    print("Prediction:")
    print(Line)
    for i in range(len(target_test)):
        print("ID: %d Expectation :%s Predicton:%s "%(i,target_test[i],ret[i]))
    print(Line)

    #print("Result is : ",ret)
    #accuracy = accuracy_score(target_test,ret)


    return (calcAccuracy(target_test,ret))

##########################################################################

#Entry Point Function
def main():
    ret = MarvellousKNN()

    print("Accuracy of KNN is : ",ret*100)

##############################################################################

#Starter
if __name__ == '__main__':
    main()
