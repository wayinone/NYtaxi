# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 12:13:44 2017

@author: Wayne
"""
from importlib import reload
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

import AuxFun
reload(AuxFun)
import logging
from sklearn.grid_search import GridSearchCV
#%%
mydf = pd.read_csv('mydf.csv')
mydf1 = mydf[mydf['outliers']==False]
mydf1 = mydf1.drop(['id','outliers'],axis=1)
#%%
z = mydf1.log_trip_duration
X = mydf1.drop(['log_trip_duration'],axis=1)
#%% For testing this file
XX = X.iloc[:500]
zz = z.iloc[:500]
#%%
test_parms = {'max_depth':[10,14,18], #maximum depth of a tree
              'eta'      :[0.025],
              'subsample':[0.8],#SGD will use this percentage of data
              'lambda '  :[3], #L2 regularization term,>1 more conservative
              'colsample_bytree ':[.8],
              'colsample_bylevel':[1],
              'min_child_weight': [5,10],
              'objective':['reg:linear'],
              'nthread'  :[-1]}
#%%

List_parm = AuxFun.genGrid(test_parms)
N = len(List_parm)
logging.basicConfig(filename='xgb_cv3.log',level=logging.DEBUG,format='%(asctime)s:%(message)s\n')
current_best_score = 999.0
Kfold = 3
count = 1

logging.info('='*40 )
#parms_df = pd.DataFrame(columns = test_parms.keys())
for i,parms in enumerate(List_parm):    
    print('%d of %d searching:' % (count,N))
    print('-'*20+'\n')    
    print(parms)
#    parms_df = pd.concat([parms_df,pd.DataFrame(parms, index=[i,])])
    logging.info('=====%d of %d searching:=====' % (count,N))
    count+=1

#    err,err_std,n_it = AuxFun.xgb_cv(parms,X,z,K = Kfold)
#    logging.info('score = %1.5f, std = %1.3f, n_boost_round =%d.'%(err,err_std,n_it))
    
    Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=i)
    #Xcv,Xv,Zcv,Zv = train_test_split(Xval, Zval, test_size=0.5, random_state=1)
    data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
    data_val  = xgb.DMatrix(Xval   , label=Zval)
    evallist = [(data_tr, 'train'), (data_val, 'valid')]

    model = xgb.train(parms, data_tr, num_boost_round=881, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=100)
    err = model.best_score
    
    ztest = model.predict(data_test)
    ytest = np.exp(ztest)-1
    
    logging.info('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))
    logging.info(parms)
    if err<current_best_score:
        current_best_score = err
        logging.info('-'*10+'Better model found')
logging.shutdown()

#%%
#cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.9],'max_depth': [3,5,8,10]}
#ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
#             'objective': 'reg:linear', 'min_child_weight': 1}
#
#GBM = GridSearchCV(xgb.XGBClassifier(**ind_params,early_stopping_rounds=30), 
#                            cv_params, 
#                             scoring = 'neg_mean_squared_error', cv = 3, n_jobs = -1)
#GBM.fit(X, z)
#%%
data_tr  = xgb.DMatrix(XX, label=zz)
parms = {'max_depth':10, #maximum depth of a tree
         'objective':'reg:linear',
         'eta'      :0.025,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :3, #L2 regularization term,>1 more conservative
         'colsample_bytree ':0.8,
         'min_child_weight': 1,
         'nthread'  :3}  #number of cpu core to use
result = xgb.cv(parms, data_tr, 
                    num_boost_round=1000,
                    early_stopping_rounds=20,# early stop if cv result is not improving
                    nfold=3,metrics="error")
#val_scores.append([result['test-error-mean'].iloc[-1],max_depth,subsample,len(result)-20])                        
#%%
List_parm = AuxFun.genGrid(test_parms)
N = len(List_parm)
#logging.basicConfig(filename='xgb_cv2.log',level=logging.DEBUG,format='%(asctime)s:%(message)s\n')
current_best_score = 999.0
Kfold = 3
count = 1
logging.info('='*40 )
for parms in List_parm:
    print('-'*40+'\n')
    print('%d of %d searching:' % (count,N))
#    logging.info('=====%d of %d searching:=====' % (count,N))
    count+=1

    err,n_it = AuxFun.xgb_cv(parms,X,z,K = Kfold)
    
#    logging.info('score = %1.5f, n_boost_round =%d.'%(err,n_it))
#    logging.info(parms)
    if err<current_best_score:
        current_best_score = err
#        logging.info('-'*10+'Better model found')
#logging.shutdown()