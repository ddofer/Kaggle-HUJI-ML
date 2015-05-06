#!/usr/bin/env python
#coding:utf-8
#Copyright (C) dirlt

import numpy as np
import datetime
import pylab as pl

def read_in(s, test = False):
    ss = s.split(',')
    (dt, tm) = ss[0].split(' ')
    (year, month, day) = map(lambda x:int(x), dt.split('-'))
    dt2 = datetime.datetime(year, month, day)
    # http://shubhamtomar.me/2015/02/25/Bike-Sharing-Demand/
    # 把weekday考虑进去
    weekday = dt2.weekday()
    hour = int(tm.split(':')[0]) # 0-23
    season = int(ss[1]) # 1-4
    holiday = int(ss[2]) # 0,1
    wkday = int(ss[3]) # 0,1
    weather = int(ss[4]) # 1-4
    temp = float(ss[5])
    atemp = float(ss[6])
    humidity = float(ss[7])
    windspd = float(ss[8])
    #    0         1       2          3        4        5    6         7     8       9       10
    # 'day'字段用来做选择validation集合
    p = [weekday, hour, year - 2011, season, holiday, wkday, weather, temp, atemp, humidity, windspd, day]
    if not test: return (p, int(ss[9]), int(ss[10]))
    else: return (p, ss[0])

def read_train():
    f = open('train.csv')
    xs = []
    ys = []
    for r in f:
        if r.startswith('date'): continue
        (x, y, y2) = read_in(r, False)
        xs.append(x)
        ys.append((y,y2))
    xs = np.array(xs)
    ys = np.array(ys)
    return (xs, ys)

def read_test():
    f = open('test.csv')
    xs = []
    dts = []
    for r in f:
        if r.startswith('date'): continue
        (x, dt) = read_in(r, True)
        xs.append(x)
        dts.append(dt)
    xs = np.array(xs)
    return (xs, dts)

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def cross_val(reg, tr):
    print 'cross validation...'
    (X, Y) = tr
    scores = []
    # kf = KFold(X.shape[0], 10)
    # for train, test in kf:
    # 选择连续日期作为测试集合比如日期为(10,11), (11,12), ... (18, 19)
    # 这样应该更加接近真实情况
    for d in range(10, 19):
        test = np.logical_or(X[:,-1] == d, X[:,-1] == (d+1))
        train = np.logical_not(test)
        (tr_x, tt_x, tr_y, tt_y) = (X[train], X[test], Y[train], Y[test])
        reg.fit(tr_x, tr_y)
        y = reg.predict(tt_x)                
        score = mean_squared_error(np.log(y + 1), np.log(np.around(tt_y[:,0] + tt_y[:,1] + 1))) ** 0.5
        print 'score = ', score
        scores.append(score)
    return np.array(scores)

class Reg:
    def __init__(self, r0, r1):
        self.r0 = r0
        self.r1 = r1
        self.select = (0,1,2,3,4,5,6,7,8,9,10)
    def fit(self, xs, ys):
        self.r0.fit(xs[:,self.select], np.log(ys[:,0] + 1))
        self.r1.fit(xs[:,self.select], np.log(ys[:,1] + 1))
    def predict(self, xs):
        ys0 = np.exp(self.r0.predict(xs[:,self.select])) - 1
        ys1 = np.exp(self.r1.predict(xs[:,self.select])) - 1
        ys = np.around(ys0 + ys1)
        ys[ys < 0] = 0
        return ys

# 不同的融合方式
class Combiner:
    def __init__(self, regs):
        self.regs = regs
    def fit(self, xs, ys):
        for r in self.regs:
            r.fit(xs, ys)
    def _predict(self, xs):
        ys0 = np.zeros(xs.shape[0])
        ys1 = np.zeros(xs.shape[0])
        for r in self.regs:
            r0 = r.r0
            r1 = r.r1
            ys0 += np.exp(r0.predict(xs[:,r.select])) - 1
            ys1 += np.exp(r1.predict(xs[:,r.select])) - 1
        ys = np.intp(np.around((ys0 + ys1) * 1.0 / len(self.regs)))
        ys[ys < 0] = 0
        return ys
    def predict(self, xs):
        ys = np.zeros(xs.shape[0])
        for r in self.regs:
            ys += r.predict(xs)
        ys *= 1.0 / len(self.regs)
        ys = np.intp(np.around(ys))
        ys[ys < 0] = 0
        return ys

def select_rf(tr):
    (X, Y) = tr
    n = 1000
    print '----- RF -----'
    # if we tune parameters.
    tuning = 0

    if not tuning:
        reg0 = RandomForestRegressor(n_estimators = n, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
        reg1 = RandomForestRegressor(n_estimators = n, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
        reg = Reg(reg0, reg1)
        return reg

    min_samples_split = [10,11,12]
    info = {}
    for mss in min_samples_split:
        print 'min_samples_split = ', mss
        reg0 = RandomForestRegressor(n_estimators = n, random_state = 0, min_samples_split = mss, oob_score = False, n_jobs = -1)
        reg1 = RandomForestRegressor(n_estimators = n, random_state = 0, min_samples_split = mss, oob_score = False, n_jobs = -1)
        reg = Reg(reg0, reg1)
        scores = cross_val(reg, tr)
        info[mss] = scores
    for mss in info:
        scores = info[mss]
        print('min_samples_split = %d, socre = %.5f(%.5f)' % (mss, scores.mean(), scores.std()))

def select_gbdt(tr):
    (X, Y) = tr
    n = 100
    print '----- GBDT -----'
    # if we tune parameters
    tuning = 0

    if not tuning:
        reg0 = GradientBoostingRegressor(n_estimators = n, max_depth = 6, random_state = 0)
        reg1 = GradientBoostingRegressor(n_estimators = n, max_depth = 6, random_state = 0)
        reg = Reg(reg0, reg1)
        return reg

    max_depth = [5,6,7]
    info = {}
    for md in max_depth:
        print 'max_depth = ', md
        reg0 = GradientBoostingRegressor(n_estimators = n, max_depth = md, random_state = 0)
        reg1 = GradientBoostingRegressor(n_estimators = n, max_depth = md, random_state = 0)
        reg = Reg(reg0, reg1)
        scores = cross_val(reg, tr)
        info[md] = scores
    for md in info:
        scores = info[md]
        print('max_depth = %d, socre = %.5f(%.5f)' % (md, scores.mean(), scores.std()))

def select():
    print('loading training set ...')
    tr = read_train()
    print('training ...')
    reg_rf = select_rf(tr)
    reg = reg_rf
    reg_gbdt = select_gbdt(tr)
    reg = reg_gbdt
    reg = Combiner([reg_rf, reg_gbdt])
    cv = 0
    if cv:
        scores = cross_val(reg, tr)
        print scores
        print scores.mean(), scores.std()
    return (reg, tr)

def run(reg, tr):
    if not reg: return
    (xs, ys) = tr
    print('training ...')
    reg.fit(xs, ys)
    print('loading test set ...')
    (xs, dts) = read_test()
    print('testing ...')
    ys = reg.predict(xs)
    f = open('submission.csv','w')
    f.write('datetime,count\n')
    n = len(dts)
    for i in xrange(0, n):
        f.write('%s,%d\n' % (dts[i], ys[i]))
    f.close()

if __name__ == '__main__':
    (reg, tr) = select()
    run(reg, tr)
