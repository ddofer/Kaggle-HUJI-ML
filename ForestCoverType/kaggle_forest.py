from sklearn import ensemble
import time

import pandas as pd
# from AdaHERF import AdaHERF
import numpy as np
from sklearn.externals import joblib
from breze.learn.lde import LinearDenoiser as lde
import os
import random
import pickle
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest

PREDICT_TEST=False


def eval_model(clf, X, y, cv=3, n_jobs=3):
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=n_jobs)
    print("cv score: {:.3f} +/-{:.3f}".format(np.mean(scores), np.std(scores)))


if __name__ == "__main__":
  loc_train = r".\kaggle_forest\train.csv"
  loc_test = r".\kaggle_forest\test.csv"
  loc_submission = r".\kaggle_forest\kaggle.forest.submission2.csv"

  df_train = pd.read_csv(loc_train)
  # df_train=df_train.astype('int32',copy=False)

  feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]
  X_train = df_train[feature_cols].values
  y = df_train['Cover_Type'].values
  print(X_train.shape)
  print("Training Data loaded")


  # pELM = Pipeline([
  #     ('aelm1', AEELMTransformer(n_components=710, activation='tanh')),
  #     # ('aelm2', AEELMTransformer(n_components=670, activation='relu')),
  #     ('aelm3', AEELMTransformer(n_components=500, activation='tanh')),
  # ])
  # p = Pipeline([
      # ('lde1', lde(p_dropout=0.15)),
      # ('lde2', lde(p_dropout=0.15)),
      # ('lde3', AEELMTransformer(n_components=670, activation='relu')),
      # ('lde4', lde(p_dropout=0.15)),
      # ('lde5', lde(p_dropout=0.15)),
  #     ('clf', ensemble.RandomForestClassifier(n_estimators = 20, n_jobs = -2)) ])

  lde_params = [0.05,0.1, 0.15]
  # lde1=lde(p_dropout=0.15)
  clf= ensemble.ExtraTreesClassifier(n_estimators = 90, n_jobs = -2)

  print("Baseline:")
  eval_model(clf, X_train, y, cv=3, n_jobs=-1)
  print("Trying linear autoencoders")
  for i in lde_params:
    lde1=lde(p_dropout=i)
    # lde1.fit(X_train)
    # X_ae = lde1.transform(X_train)
    for j in lde_params:
      lde2=lde(p_dropout=j)
      # lde2.fit(X_ae)
      # X_ae = lde2.transform(X_ae)
      # # lde3=lde(p_dropout=0.15)
      print("For p_dropout = ",i , j)
      # eval_model(clf, X_ae, y, cv=3, n_jobs=-1)
      univ_selection=SelectKBest(k=52)
      combined_features = FeatureUnion([("lde1", lde1), ("lde2", lde2),("univ_selection",univ_selection)])
      # Use combined features to transform dataset:
      combined_features.fit(X_train, y)
      X_features = combined_features.transform(X_train)
      eval_model(clf, X_features, y, cv=3, n_jobs=-1)


  print("linear AE done")

  # ae_params = {'p_dropout':[0.1, 0.3]}
  # estimator = GridSearchCV(p, cv = 2,
  #                        # param_grid = dict(lde1__p_dropout = lde_params,lde2__p_dropout = lde_params)
  #                        )

  # estimator.fit(X_train, y)
  # print("best_estimator_ ",estimator.best_estimator_ )
  # print("best_score_  ",estimator.best_score_ )


  try:
      clf = joblib.load('jlib.pkl')
      print("pickled model loaded")
  except:
      # clf =  AdaHERF()
      clf = ensemble.ExtraTreesClassifier(n_estimators = 80, n_jobs = -2)
      clf.fit(X_train, y)
      print("Fitted")
      joblib.dump(clf, 'jlib.pkl')
      print("joblib pickled")


if PREDICT_TEST is True:
    df_test = pd.read_csv(loc_test)
    df_test=df_test.astype('int32',copy=False)
    X_test = df_test[feature_cols].values
    test_ids = df_test['Id'].values
    print("Test Data loaded")

    with open(loc_submission, "wt") as outfile:
      outfile.write("Id,Cover_Type\n")
      # outfile.write("")
      for e, val in enumerate(list(clf.predict(X_test))): #Change this to iterate/yield over each sample. (mem issues)
        outfile.write("%s,%s\n"%(test_ids[e],val)) #Check why val is printed as "[1.]" rather than "1"
      print("Done")