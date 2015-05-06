from sklearn import ensemble

import pandas as pd
# from AdaHERF import AdaHERF
import numpy as np
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import random
import pickle
import neurolab as nl

from TestModels import CV_Binary_stats

'''
http://pythonhosted.org/neurolab/ex_newff.html
https://code.google.com/p/neurolab/
http://stackoverflow.com/questions/24145006/neurolab-predicted-values-for-test-set
'''

if __name__ == "__main__":
    loc_train = r".\kaggle_forest\train.csv"
    loc_test = r".\kaggle_forest\test.csv"


    df_train = pd.read_csv(loc_train)
    #  df_test = pd.read_csv(loc_test)
    #  df_test=df_test.astype('int32',copy=False)

    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

    X_train = df_train[feature_cols].values
    # X_test = df_test[feature_cols].values
    y = df_train['Cover_Type'].values
    # test_ids = df_test['Id'].values

    x=X_train
    print("Data loaded")


    MM_scaler = MinMaxScaler(feature_range=(-10, 10), copy=False)
    Scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
    x = Scaler.fit_transform(x.astype(np.float))
    x = MM_scaler.fit_transform(x.astype(np.float))


    # http://pythonhosted.org/neurolab/ex_newff.html

    # Create train samples
    # x = np.linspace(-7, 7, 20)
    # y = np.sin(x) * 0.5

    size = len(x)

    # inp = x.reshape(size,1) #Orig
    # tar = y.reshape(size,1) #Orig
    inp = x
    tar = y

    # Create network with 2 layers and random initialized
    # net = nl.net.newff([[-7, 7]],[5, 1]) #Orig

    # Create network with layers..
    # net = nl.net.newff([[-7, 7]],[54,110, 7])
    net = nl.net.newff(minmax=[-10, 10],size=[len(x),254,50, 7])

    # Train network
    error = net.train(inp, tar, epochs=10, show=10, goal=0.05)

    # Simulate network
    out = net.sim(inp)

    # Plot result
    import pylab as pl
    pl.subplot(211)
    pl.plot(error)
    pl.xlabel('Epoch number')
    pl.ylabel('error (default SSE)')

    # x2 = np.linspace(-6.0,6.0,150)
    # y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)

    y3 = out.reshape(size)

    # pl.subplot(212)
    # pl.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
    # pl.legend(['train target', 'net output'])
    # pl.show()



