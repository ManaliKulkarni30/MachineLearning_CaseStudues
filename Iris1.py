###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:28th March 2021
#About: Loading iris dataset using inbuilt function
#
###########################################################################

#Required import
from sklearn.datasets import load_iris

############################################################################

#Entry poitn Function
def main():
    #Loading Dataset
    dataset = load_iris()

    print("Features of Dataset: ")
    print(dataset.feature_names)

    print("Target of Dataset: ")
    print(dataset.target_names)

##############################################################################

#Starter
if __name__ == '__main__':
    main()
