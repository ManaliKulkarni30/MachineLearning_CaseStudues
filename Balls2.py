###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:28th March 2021
#About: Implementing Decision Tree on Demo Dataset
#Note: This file removes the error of Balls1 program
#
###########################################################################

#Required import
from sklearn import tree

###########################################################################

#Enrty poitn function
def main():
    #Step 1 & 2
    #rought = 1, smooth = 0
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],
    [96,0],[43,1],[110,0],[35,1],[95,0]]

    #1: Tennis 2:Cricket
    Labele = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    #Step 3
    dobj = tree.DecisionTreeClassifier()

    #Step 4
    dobj.fit(Features,Labele)

    #Step 5
    result = dobj.predict([[40,1]])

    print("Ball is ",result)

################################################################################

#Starter
if __name__ == '__main__':
    main()
