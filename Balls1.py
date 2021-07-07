###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:28th March 2021
#About: Implementing Decision Tree on Demo Dataset
#Note: This code contains error since the fit function only accepts integer or float and here we are passing
#      list containing string which is not allowed
#
###########################################################################

#Required import
from sklearn import tree

############################################################################

#Entry Point Function
def main():
    #Step 1 & 2
    Features = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],[48,"Rough"],[90,"Smooth"],
    [35,"Rough"],[92,"Smooth"],[35,"Rough"],[35,"Rough"],[35,"Rough"],
    [96,"Smooth"],[43,"Rough"],[110,"Smooth"],[35,"Rough"],[95,"Smooth"]]

    Labele = ["Tennis","Tennis","Cricket","Tennis","Cricket",
     "Tennis","Cricket","Tennis","Tennis","Tennis",
     "Cricket","Tennis","Cricket","Tennis","Cricket"]

    dobj = tree.DecisionTreeClassifier()

    #Step 4
    dobj.fit(Features,Labele)#Error

    #Step 5
    result = dobj.predict([[40,1]])

    print("Ball is ",result)

###############################################################################

#Starter
if __name__ == '__main__':
    main()
