
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import time
import math
from math import sqrt
import scipy
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from keras.models import Sequential,Model
import keras
from keras.layers import Dense, Dropout, SpatialDropout2D, Flatten, Activation, merge, Input, Masking
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit,train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import svm
from math import sqrt
from sklearn.metrics import mean_squared_error,r2_score
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler


# In[2]:

mode='interactive'


# In[3]:

if mode == 'interactive':
    taskname = 'il'
    pathname = 'chemception'
elif mode == 'batch':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='IL')
    parser.add_argument('task')
    parser.parse_args(sys.argv[1:])


# In[4]:

if mode == "interactive":
    archdir = "archive/archive/"
elif mode == "batch":
    archdir = "../../archive/archive/"
    
jobname='il'
taskname='descriptors'


# In[5]:

filename=str(archdir+"training_features.npy")
X_train=np.load(filename)
filename=str(archdir+"testing_features.npy")
X_test=np.load(filename)
filename=str(archdir+"training_target.npy")
y_train=np.load(filename)
filename=str(archdir+"testing_target.npy")
y_test=np.load(filename)  


# In[6]:

def prep_X(array):                        #tabular data 
    X_sample=array.astype('float32')
    print(X_sample.shape)
    return X_sample


# In[7]:

def prep_Y(array):
    y_sample = array.astype("float32")
    print(y_sample.shape)
    return y_sample


# In[8]:

X_train = prep_X(X_train)
X_test = prep_X(X_test)
y_train = prep_Y(y_train)
y_test = prep_Y(y_test)


# In[9]:

n_fold=5
epoch=500
batch_size=30
np.random.seed(7)


# In[10]:

def rmse(y,y_pred):
    rms=sqrt(mean_squared_error(y,y_pred))
    return rms


# In[11]:

def aard(y,y_pred):
    dev=abs((y-y_pred)/(y))*100
    return np.mean(dev)


# In[12]:

def avg(x):
    return np.mean(x)


# def trees(n_estimators):
#     rf=RandomForestRegressor(criterion='mse',n_estimators=n_estimators,random_state=7)
#     return rf

# In[13]:

def setup_mlp(params):
    mlp_input = Input(shape=(int(X_train.shape[1]),))
    x = Dense(params['mlp1_units'],init='glorot_normal', activation="relu")(mlp_input)
    x = Dropout(0.5)(x)
    x = Dense(params['mlp2_units'],init='glorot_normal', activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1,init='glorot_normal',activation='linear')(x)
    
    model = Model(mlp_input, x)
    print(model.summary())
    model.compile(optimizer="adam", loss="mean_squared_error",metrics=['mse'])
    return model
   


# In[14]:

def final(n_fold,X_train,y_train,y_test):
    kf = KFold(n_splits=n_fold, shuffle=False, random_state=7) 
    fold=0
    cv_train_rmse=[]
    cv_train_r2=[]
    cv_train_aard=[]

    cv_val_rmse=[]
    cv_val_r2=[]
    cv_val_aard=[]
      
    cv_test_rmse=[]
    cv_test_r2=[]
    cv_test_aard=[]
    for train_index, test_index in kf.split(X_train):
        fold=fold+1
        print('fold: '+str(fold))
        X_trainn, X_val = X_train[train_index], X_train[test_index]
        y_trainn, y_val = y_train[train_index], y_train[test_index]
        print('Train set: '+str(X_trainn.shape)+str(y_trainn.shape))
        print('Validation set: '+str(X_val.shape)+str(y_val.shape))
        print('Test set: '+str(X_test.shape)+str(y_test.shape))
        
        model=setup_mlp(params)
        print(model.summary())
        #model.fit(X_trainn,y_trainn) #check model.fit for random forest
        
        model_json = model.to_json()
        filepath=str(taskname+"_")+str("architecture_")+str(fold)+str(".json")
        with open(filepath, "w") as json_file:
            json_file.write(model_json)
        
        early = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        saveBestModel = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
        
        model.fit(X_trainn,y_trainn,validation_data=(X_val,y_val),nb_epoch=epoch,batch_size=batch_size,callbacks=[early,saveBestModel],verbose=2)
        
        
        y_pred_train=model.predict(X_trainn)
        y_pred_val=model.predict(X_val)
        y_pred_test=model.predict(X_test)
        
        cv_train_rmse.append(rmse(y_trainn,y_pred_train))
        cv_train_r2.append(r2_score(y_trainn,y_pred_train))
        cv_train_aard.append(aard(y_trainn,y_pred_train))
    
        cv_val_rmse.append(rmse(y_val,y_pred_val))
        cv_val_r2.append(r2_score(y_val,y_pred_val))
        cv_val_aard.append(aard(y_val,y_pred_val))
    
        cv_test_rmse.append(rmse(y_test,y_pred_test))
        cv_test_r2.append(r2_score(y_test,y_pred_test))
        cv_test_aard.append(aard(y_test,y_pred_test))
    cv_train_rmse=avg([cv_train_rmse])
    cv_train_r2=avg([cv_train_r2])
    cv_train_aard=avg([cv_train_aard])
    cv_val_rmse=avg([cv_val_rmse])
    cv_val_r2=avg([cv_val_r2])
    cv_val_aard=avg([cv_val_aard])
    cv_test_rmse=avg([cv_test_rmse]) 
    cv_test_r2=avg([cv_test_r2])
    cv_test_aard=avg([cv_test_aard])
    return cv_train_rmse,cv_train_r2,cv_train_aard,cv_val_rmse, cv_val_r2, cv_val_aard, cv_test_rmse, cv_test_r2, cv_test_aard


# final(n_fold,X_train,y_train,y_test)

# In[ ]:

params= {"mlp1_units":16, "mlp2_units":16}
final(n_fold,X_train,y_train,y_test)
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":32, "mlp2_units":32}
final(n_fold,X_train,y_train,y_test)
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":64, "mlp2_units":64}
final(n_fold,X_train,y_train,y_test)
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":128, "mlp2_units":128}
final(n_fold,X_train,y_train,y_test)
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":256, "mlp2_units":256}
final(n_fold,X_train,y_train,y_test)
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_train,y_train,y_test)))


# In[ ]:



