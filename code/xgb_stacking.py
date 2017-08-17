# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:31:44 2017

@author: Wayne
"""
from importlib import reload
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import AuxFun
reload(AuxFun)
import logging
import time
import pickle
#%% Kick off outliers
mydf1= mydf[outliers.outliers==False]
z = np.log(data[outliers.outliers==False].trip_duration+1)
X = mydf1
data_test = xgb.DMatrix(testdf)
#%% Without using outliers
z = np.log(data.trip_duration+1)
X = mydf
data_test = xgb.DMatrix(testdf)
#%%
test_parms = {'max_depth':[14,14,14,14], #maximum depth of a tree
              'eta'      :[0.025],
              'subsample':[0.8],#SGD will use this percentage of data
              'lambda '  :[3], #L2 regularization term,>1 more conservative
              'colsample_bytree ':[.8],
              'colsample_bylevel':[1],
              'min_child_weight': [10],
              'objective':['reg:linear'],
              'nthread'  :[-1]}
#%%
if np.all(X.keys()==testdf.keys()):
    print('Good! The keys of training feature is identical to those of test feature.')
    print('They both have %d features, as follows:'%len(X.keys()))
    print(list(X.keys()))
else:
    print('Oops, something is wrong, keys in training and testing are not matching')
   
#%%
XX = X.iloc[:500]
zz = z.iloc[:500]
#%%
List_parm = AuxFun.genGrid(test_parms)
N = len(List_parm)
logging.basicConfig(filename='xgb_cv3.log',level=logging.DEBUG,format='%(asctime)s:%(message)s\n')
#%%
ztest_mat = np.zeros((len(testdf),N))

#%%
logging.info('='*40 )
#parms_df = pd.DataFrame(columns = test_parms.keys())
count = 1
for i,parms in enumerate(List_parm):    
    print('%d of %d searching:' % (count,N))
    print('-'*20+'\n')    
    print(parms)
    logging.info('=====%d of %d searching:=====' % (count,N))
    count+=1
    tic = time.time()
    
    Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=i+15)
    
    data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
    data_val  = xgb.DMatrix(Xval   , label=Zval)
    evallist = [(data_tr, 'train'), (data_val, 'valid')]

    model = xgb.train(parms, data_tr, num_boost_round=2000, evals = evallist,
                  early_stopping_rounds=30, maximize=False, verbose_eval=100)
    err = model.best_score
    
    ztest_mat[:,i] = model.predict(data_test)
    
    print('score = %1.5f, n_boost_round =%d, took %d second'
          %(err,model.best_iteration,round(time.time()-tic)))

    logging.info('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))
    logging.info(parms)

logging.shutdown()
#%%
with open('ztest_mat_kmeans10.pickle', "wb") as output_file:
    pickle.dump(ztest_mat, output_file)
#%%
#%% 1. avg before exp 0.379
ztest = np.mean(ztest_mat,axis=1)
ytest = np.exp(ztest)-1
submission = pd.DataFrame({'id': test.id, 'trip_duration': ytest})
submission.to_csv('submission13.csv', index=False)
#%% 2. avg after exp
ytest_mat = np.exp(ztest_mat)-1
ytest = np.mean(ytest_mat,axis=1)                 
submission = pd.DataFrame({'id': test.id, 'trip_duration': ytest})
submission.to_csv('submission11.csv', index=False)
#%%

with open('ztest_mat2.pickle', "rb") as input_file:
    ttt = pickle.load(input_file)
#%%
xgb.plot_importance(model)
#%%
ztest_mat = np.hstack((ztest_mat,ttt))