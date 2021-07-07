###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:28th March 2021
#About: Implementing Decision Tree on Demo Dataset
#Note: This is sytematic code which uses main function,a machine Learning function and starter code
#
###########################################################################

#Required import
from sklearn import tree

###########################################################################

#Helper Function
def MarvellousMl(weight,surface):
    #Step 1 n 2
    #Rough : 1 Smooth:0
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],
    [96,0],[43,1],[110,0],[35,1],[95,0]]

    #1: Tennis 2:Cricket
    Labele = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    #Step 3
    dobj = tree.DecisionTreeClassifier()

    #Step 4
    dobj.fit(Features,Labele)

    #Step 5
    result = dobj.predict([[weight,surface]])

    if result == 1:
        print("Your object looks like Tennis Ball")
    else:
        print("Your object looks like Cricket Ball")

############################################################################

#Entry Point Function
def main():
    print("-------------------------Supervised Machine Learning----------------------------")
    weight = int(input("Enter weight: "))
    surface = input("Enter surface: ")

    #For programmer convinience converting all input to lowercase so that program won't get confused
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Invalid Input")
        return

    MarvellousMl(weight,surface)

###########################################################################

#Starter
if __name__ == '__main__':
    main()
