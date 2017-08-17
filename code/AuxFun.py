# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 00:17:23 2017

@author: Wayne
"""

from datetime import date
import holidays
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb
import time
import numpy as np
#%%
def datetime2num(datetimes):
    '''
    Input:
        datetimes: list of datetimes, with format like
            '2016-03-14 17:24:55'
    Output:
        isholiday: 1 if holiday, 0 if not
        hour: the hour of the day. 
            e.g. 7.5 means 7:30 am
                18.75 means 6:45 pm
    '''
    us_holidays = holidays.UnitedStates()
    isholiday = [None]*len(datetimes)
    hour = [None]*len(datetimes)
    for i,c in enumerate(datetimes):
        cs = c.split(' ')
        day = list(map(int,cs[0].split('-')))
        isholiday[i]=date(*day).isoweekday() in (6,7) or date(*day) in us_holidays 
        time= list(map(float,cs[1].split(':')))
        hour[i]=time[0]+time[1]/60+time[2]/3600
    return isholiday,hour
#%%
def xgb_cv(parms,X,y,K=4,test_size =.2, random_seed = 1):
    '''
    X: pandas DataFrame
    y: pandas DataFrame
    '''
    
    rs = ShuffleSplit(n_splits=K, test_size=test_size, random_state=random_seed)
    best_score = []
    best_iteration=[]
    k=1
    for train_index, test_index in rs.split(X):
        print('running %d / %d cv...\n' %(k,K))
        k+=1
        tic = time.time()
        
        Xtrain = X.iloc[train_index]
        ytrain = y.iloc[train_index]
        Xval   = X.iloc[test_index]
        yval   = y.iloc[test_index]
        
        data_tr  = xgb.DMatrix(Xtrain, label=ytrain)
        data_val = xgb.DMatrix(Xval   , label=yval)
        evallist = [(data_tr, 'train'), (data_val, 'valid')]
        
        xgbmodel = xgb.train(parms, data_tr, num_boost_round=2000, evals = evallist,
                          early_stopping_rounds=30, maximize=False, 
                          verbose_eval=200)
        best_score.append(xgbmodel.best_score)  
        best_iteration.append(xgbmodel.best_iteration)
        print('    -score =',xgbmodel.best_score,', took %d sec\n' % (time.time()-tic))
    scores = np.array(best_score)    
    return scores.mean(),scores.std(),sum(best_iteration)//K
    
#%% The following generate K-fold cv sample index
'''
K=3
rs = ShuffleSplit(n_splits=K, test_size=.25, random_state=0)
X = [None]*10
for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
'''
#%%
def Lists_combination(ListofList):
    '''
    Generate all the combination of lists:
        e.g.
            A =[1,2,3]; B = ['a','b']
            C = Lists_combination([A,B])
            C = [[1, 'a'], [1, 'b'], [2, 'a'], [2, 'b'], [3, 'a'], [3, 'b']]
    '''
    
    ans = []
    def dfs(L,record):
        if not L:
            ans.append(record)
            return
        C = L[0]
        for c in C:
            dfs(L[1:],record+[c])
    dfs(ListofList,[])
    return ans
#%%
def genGrid(test_parms,shuffle=True):
    '''
    Input: dictionary, where all the values are list
    e.g.
        test_parms = {'max_depth':[8,10,14],
                      'eta'      :[0.2]}
    Return: List of dictionary.
    e.g. 
        [{'eta': 0.2, 'max_depth': 8},
         {'eta': 0.2, 'max_depth': 10},
         {'eta': 0.2, 'max_depth': 14}]
    '''
    D = Lists_combination(list(test_parms.values()))
    Key = list(test_parms.keys())
    Listdic = []
    for i,a in enumerate(D):
        parms_i = {}
        for j,k in enumerate(Key):
            parms_i[k] = D[i][j]
        Listdic.append(parms_i)
    if shuffle:
        np.random.shuffle(Listdic)
    return Listdic