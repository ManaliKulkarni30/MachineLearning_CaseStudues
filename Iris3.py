###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:28th March 2021
#About: Applying Decision Tree Algorithm on Iris Dataset
#
###########################################################################

#Required imports
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

##########################################################################

#Entry Point Function
def main():
    #Loading Dataset
    dataset = load_iris()

    print("Features of Dataset: ")
    print(dataset.feature_names)

    print("Target of Dataset: ")
    print(dataset.target_names)

    #print("Iris Dataset is: ")
    #for iCnt in range(len(dataset.target)):
        #print("ID : %d Features:%s Lable: %s"%(iCnt,dataset.data[iCnt],dataset.target[iCnt]))

    index = [1,51,101]
    test_target = dataset.target[index]
    test_feature = dataset.data[index]

    train_target = np.delete(dataset.target,index)
    train_feature = np.delete(dataset.data,index,axis = 0)

    obj = tree.DecisionTreeClassifier()

    obj.fit(train_feature,train_target)

    result = obj.predict(test_feature)

    print("Result predicted by ML: ",result)
    print("Result expected: ",test_target)

##############################################################################

#Starter
if __name__ == '__main__':
    main()
