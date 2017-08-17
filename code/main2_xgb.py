# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 00:25:27 2017

@author: Wayne
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
#%%
mydf1= mydf[outliers.outliers==False]
z = np.log(data.trip_duration+1)
X = mydf1
Xtest = testdf
data_test = xgb.DMatrix(Xtest)
#%%
rmse = lambda z,zp:np.sqrt(np.mean((z-zp)**2))
#%%

parms = {'max_depth':14, #maximum depth of a tree
         'objective':'reg:linear',
         'eta'      :0.025,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10,
         'nthread'  :3}  #number of cpu core to use
#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=1)
#Xcv,Xv,Zcv,Zv = train_test_split(Xval, Zval, test_size=0.5, random_state=1)
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_val  = xgb.DMatrix(Xval   , label=Zval)
evallist = [(data_tr, 'train'), (data_val, 'valid')]

model = xgb.train(parms, data_tr, num_boost_round=881, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=100)

print('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))
#%% training all the data
data_train  = xgb.DMatrix(X, label=z)
evallist = [(data_train, 'train')]

model = xgb.train(parms, data_train, num_boost_round=880, evals = evallist,
                  maximize=False, 
                  verbose_eval=100)
#%%


#%%
ztest = model.predict(data_test)
#%%
ytest = np.exp(ztest)-1
submission = pd.DataFrame({'id': test.id, 'trip_duration': ytest})
submission.to_csv('submission_1.csv', index=False)
#%%
with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
#%%
for d in (mydf,testdf):
    print(d.Temp.mean())
#%%
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('oops')
print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values))== 0 else print('oops')
print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] and test.count().min() == test.shape[0] else print('oops')
print('The store_and_fwd_flag has only two values {}.'.format(str(set(train.store_and_fwd_flag.unique()) | set(test.store_and_fwd_flag.unique()))))
#%% Kmeans
from sklearn.cluster import MiniBatchKMeans
coords = np.vstack((mydf[['pickup_latitude', 'pickup_longitude']].values,
                    mydf[['dropoff_latitude', 'dropoff_longitude']].values,
                    testdf[['pickup_latitude', 'pickup_longitude']].values,
                    testdf[['dropoff_latitude', 'dropoff_longitude']].values))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=20, batch_size=10000).fit(coords[sample_ind])
for df in (mydf,testdf):
    df.loc[:, 'pickup_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df.loc[:, 'dropoff_loc'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
#%%
train_loc = [None]*2;test_loc=[None]*2
for i,loc in enumerate(['pickup_loc','dropoff_loc']):
    train_loc[i]= pd.get_dummies(mydf[loc], prefix=loc, prefix_sep='_')    
    test_loc[i] = pd.get_dummies(testdf[loc], prefix=loc, prefix_sep='_')
train_loc = pd.concat(train_loc,axis=1)
test_loc  = pd.concat(test_loc,axis=1)

#%%
mydf1 = pd.concat([mydf,train_loc],axis = 1)
testdf1 = pd.concat([testdf,test_loc],axis = 1)
#%%
mydf1 = mydf1[mydf1['outliers']==False]
mydf1 = mydf1.drop(['id','outliers'],axis=1)
z = mydf1.log_trip_duration
X = mydf1.drop(['log_trip_duration'],axis=1)
Xtest = testdf1.drop('id',axis=1)
#%%
X = X.drop(['pickup_loc','dropoff_loc'],axis=1)
#%%
Xtest=Xtest.drop(['pickup_loc','dropoff_loc'],axis=1)