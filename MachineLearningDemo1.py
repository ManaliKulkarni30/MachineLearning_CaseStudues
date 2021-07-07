###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:4th April 2021
#About: Applying Decision Tree Algorithm on Iris Dataset and calculating Accuracy
#
###########################################################################

#Required imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

###########################################################################

#Helper Function (Decision Tree)
def MarvellousDicision():
    dataset = load_iris()

    data = dataset.data
    target = dataset.target

    data_train,data_test,targe_train,target_test = train_test_split(data,target,test_size=0.5)

    cobj = tree.DecisionTreeClassifier()

    cobj.fit(data_train,targe_train)

    output = cobj.predict(data_test)

    Accuracy = accuracy_score(target_test,output)#IMP
    #syntax : acuracy_score(expected kay hota,kay o/p ala)

    return Accuracy

##########################################################################

#Entry Point Function
def main():
    ret = MarvellousDicision()
    print("Accuracy of Decision Tree Algorithm is : ",ret*100,"%")

##############################################################################

#Starter
if __name__ == '__main__':
    main()
