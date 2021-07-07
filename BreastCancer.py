###########################################################################
#
#Author:Manali Milind Kulkarni
#Date:22nd June 2021
#About: Implementing Random Forest Classifier to predict the breast cancer.
#
###########################################################################

#Required Python Package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pdb

##########################################################################

#File Path
INPUT_PATH = "breast-cancer-wisconsin.csv"
OUTPUT_PATH = "data.csv"

#Headers
HEADERS = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape",
"MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli",
"Mitoses","CancerType"]

##########################################################################

def get_headrs(dataset):
    """
    dataset headers
    :param dataset:
    return
    """
    return dataset.columns.values

###########################################################################

def add_headers(dataset,header):
    """
    Add the headers to the dataset
    :param dataset:
    :param header:
    :return:
    """
    dataset.columns = header
    return dataset

##########################################################################

def data_file_to_csv():
    """
    :return:
    """

    headers = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape",
    "MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli",
    "Mitoses","CancerType"]
    #Load the dataset into pandas dataframe
    dataset = pd.read_csv(INPUT_PATH)
    #add the headers to the loaded dataset
    dataset = add_headers(dataset,headers)
    #Save the loaded datset into CSV Format
    dataset.to_csv(OUTPUT_PATH,index=False)

##########################################################################

def split_dataset(dataset,train_percentage,feature_headers,target_header):
    """
    Split the dataset with train percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x,train_y, target_x, target_y
    """
    train_x,test_x,train_y,test_y = train_test_split(dataset[feature_headers],dataset[target_header],train_size=train_percentage)
    return train_x,test_x,train_y,test_y

########################################################################

def handel_missing_values(dataset,missing_values_header,missing_label):
    """
    Filter missing values from dataset
    :param dataset:
    :param missing_values_header:
    :param missing_label:
    :return:
    """
    return dataset[dataset[missing_values_header]!=missing_label]

######################################################################

def random_forest_classifier(features,target):
    """
    To train the random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features,target)
    return clf

#####################################################################

def datset_statistics(dataset):
    print(dataset.describe())

#####################################################################

def main():
    data_file_to_csv()
    #Load the csv file into pandas dataframe
    dataset=pd.read_csv(OUTPUT_PATH)
    #Get the statistics of the dataset
    datset_statistics(dataset)

    #Filter missing values
    dataset = handel_missing_values(dataset,HEADERS[6],'?')
    train_x,test_x,train_y,test_y = split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])

    #Train and Test dataset size details
    print("Train_X shape : ",train_x.shape)
    print("Train_Y shape : ",train_y.shape)
    print("Test_X shape : ",test_x.shape)
    print("Test_Y shape : ",test_y.shape)

    #Create random forest classifier instances
    trained_model = random_forest_classifier(train_x,train_y)
    print("Trained Model :: ",trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(0,5):
        print("Actual Outcome :: {} Predicted Output :: {}".format(list(test_y)[i],predictions[i]))

    print("Train Accuracy :: ",accuracy_score(train_y,trained_model.predict(train_x))*100)
    print("Test Accuracy :: ",accuracy_score(test_y,predictions)*100)
    print("Confusion Matrix :: ")
    print(confusion_matrix(test_y,predictions))

#############################################################################

if __name__ == '__main__':
    print("-------------------------Breast Cancer Prediction-----------------------------")
    main()

#############################################################################
# OUTPUT:
"""
D:\Python\CaseStudies>BreastCancer.py
-------------------------Breast Cancer Prediction-----------------------------
         CodeNumber  ClumpThickness  UniformityCellSize  UniformityCellShape  ...  BlandChromatin  NormalNucleoli     Mitoses  CancerType
count  6.980000e+02      698.000000          698.000000           698.000000  ...      698.000000      698.000000  698.000000  698.000000
mean   1.071807e+06        4.416905            3.137536             3.210602  ...        3.438395        2.869628    1.590258    2.690544
std    6.175323e+05        2.817673            3.052575             2.972867  ...        2.440056        3.055004    1.716162    0.951596
min    6.163400e+04        1.000000            1.000000             1.000000  ...        1.000000        1.000000    1.000000    2.000000
25%    8.702582e+05        2.000000            1.000000             1.000000  ...        2.000000        1.000000    1.000000    2.000000
50%    1.171710e+06        4.000000            1.000000             1.000000  ...        3.000000        1.000000    1.000000    2.000000
75%    1.238354e+06        6.000000            5.000000             5.000000  ...        5.000000        4.000000    1.000000    4.000000
max    1.345435e+07       10.000000           10.000000            10.000000  ...       10.000000       10.000000   10.000000    4.000000

[8 rows x 10 columns]
Train_X shape :  (477, 9)
Train_Y shape :  (477,)
Test_X shape :  (205, 9)
Test_Y shape :  (205,)
Trained Model ::  RandomForestClassifier()
Actual Outcome :: 2 Predicted Output :: 2
Actual Outcome :: 4 Predicted Output :: 4
Actual Outcome :: 2 Predicted Output :: 2
Actual Outcome :: 4 Predicted Output :: 4
Actual Outcome :: 2 Predicted Output :: 2
Train Accuracy ::  100.0
Test Accuracy ::  97.5609756097561
Confusion Matrix ::
[[137   3]
 [  2  63]]"""
 ###############################################################################
