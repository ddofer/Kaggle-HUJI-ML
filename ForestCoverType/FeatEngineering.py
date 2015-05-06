'''
http://nbviewer.ipython.org/github/aguschin/kaggle/blob/master/forestCoverType_featuresEngineering.ipynb
'''
import numpy as np
import pandas as pd
from sklearn import ensemble
# %matplotlib inline
# import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  # loc_train = "kaggle_forest\\train.csv"
  # loc_test = "kaggle_forest\\test.csv"


  # df_train = pd.read_csv(loc_train)
  # df_test = pd.read_csv(loc_test)

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    def r(x):
        if x+180>360:
            return x-180
        else:
            return x+180

    train['Aspect2'] = train.Aspect.map(r)
    test['Aspect2'] = test.Aspect.map(r)
    train['Highwater'] = train.Vertical_Distance_To_Hydrology < 0
    test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0
    train['EVDtH'] = train.Elevation-train.Vertical_Distance_To_Hydrology
    test['EVDtH'] = test.Elevation-test.Vertical_Distance_To_Hydrology

    train['EHDtH'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2
    test['EHDtH'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2

    train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
    test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5

    train['Hydro_Fire_1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
    test['Hydro_Fire_1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']

    train['Hydro_Fire_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
    test['Hydro_Fire_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

    train['Hydro_Road_1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
    test['Hydro_Road_1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

    train['Hydro_Road_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
    test['Hydro_Road_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

    train['Fire_Road_1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
    test['Fire_Road_1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

    train['Fire_Road_2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
    test['Fire_Road_2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

#remove inddex when saving
    train.to_csv('n_train.csv')
    test.to_csv('n_test.csv')