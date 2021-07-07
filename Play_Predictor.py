#1d array is considered as series
#2d array is known as Dataframe
##d array is known as pannel which is depricated from python 3
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def MarvellousPedictior(path):
    #Step 1
    #Dataframe
    df = pd.read_csv(path)
    print("Dataset size is : ",len(df))

    #Step 2
    Features = ["Wether","Temperature"]

    Wether = df.Wether
    Temperature = df.Temperature
    Play = df.Play

    le = LabelEncoder()

    #series
    WetherX = le.fit_transform(Wether)
    TemperatureX = le.fit_transform(Temperature)
    Label = le.fit_transform(Play)

    features = list(zip(WetherX,TemperatureX))
    #Step 3
    cobj = KNeighborsClassifier(n_neighbors=3)

    cobj.fit(features,Label)

    #Step 4
    output = cobj.predict([[0,2]])

    if output == 1:
        print("You can play")
    else:
        print("You can not Play")

def main():
     print("--------------------Marvellous Play Predictor-------------------------")
     path = input("Enter path of file : ")

     MarvellousPedictior(path)
if __name__ == '__main__':
    main()
