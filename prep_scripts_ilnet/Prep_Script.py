
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
from math import log


# In[2]:

def load_data(csv,drop_columns,target_column):
    df=pd.read_csv(csv)
    df_output=df[target_column]
    if type(target_column) ==list:
        for i in df_output.columns:
            zero=df_output[df_output[i]==0]
            print('There are '+str(zero.shape[0])+' zero target values')
            df=df.drop(df.index[zero.index.values])
            print('Dropping '+str(zero.shape[0])+' row')
    else:
        zero=df[df[target_column]==0]
        print('There are '+str(zero.shape[0])+' zero target values')
        df=df.drop(df.index[zero.index.values])
        print('Dropping '+str(zero.shape[0])+' row')
        
    print(df.isnull().values.any())
    nan_rows = df[df.isnull().T.any().T]
    print('Removing '+str(nan_rows.shape[0]) +' nan rows')
    df=df.dropna()
    print(df.shape)
    if drop_columns==None:
        pass
    else:
        df=df.drop(drop_columns,axis=1)
    Y=df[target_column]
    Y=np.array(Y)
    Y[Y != 0]
    if type(target_column) ==list:
        df=df.drop(target_column,axis=1)
    else:
        df=df.drop([target_column],axis=1)
    X=np.array(df)
    print('Features shape: '+str(X.shape))
    print('Target shape: '+str(Y.shape))
    return X,Y


# In[3]:

def data_split(features,target):
    split=train_test_split(features,target,test_size=0.25,random_state=7)
    X_train=split[0]
    X_test=split[1]
    y_train=split[2]
    y_test=split[3]
    print('X_train shape: '+str(X_train.shape))
    print('y_train shape: '+str(y_train.shape))
    print('X_test shape: '+str(X_test.shape))
    print('y_test shape: '+str(y_test.shape))
    return X_train, X_test, y_train, y_test 


# In[4]:

def normalize_data(features_train,features_test):
    scaler=StandardScaler()
    X_trained=scaler.fit_transform(features_train)
    X_tested=scaler.transform(features_test)
    print('X_train_normalize shape: '+str(X_trained.shape))
    print('X_test_normalize shape: '+str(X_tested.shape))
    return X_trained,X_tested


# In[5]:

def log_output(target_train,target_test):
    y_trained=np.log(target_train)
    y_tested=np.log(target_test)
    print('y_train_log shape: '+str(y_trained.shape))
    print('y_test_log shape: '+str(y_tested.shape))
    return y_trained,y_tested


# In[6]:

X,Y=load_data('salts+descriptors+all.csv',['salt_name','name-cation','name-anion','Unnamed: 0','Unnamed: 0.1','smiles-cation','smiles-anion','Specific density, kg/m<SUP>3</SUP>','Viscosity, Pa&#8226;s'],          'Heat capacity at constant pressure, J/K/mol')
X_train, X_test, y_train, y_test = data_split(X,Y)
X_trained,X_tested= normalize_data(X_train, X_test)
y_trained,y_tested= log_output(y_train,y_test)
np.save('cpt_training_features.npy',X_trained)
np.save('cpt_testing_features.npy',X_tested)
np.save('cpt_training_target.npy',y_trained)
np.save('cpt_testing_target.npy',y_tested)


# In[7]:

X,Y=load_data('salts+descriptors+all.csv',['salt_name','name-cation','name-anion','Unnamed: 0','Unnamed: 0.1','smiles-cation','smiles-anion','Viscosity, Pa&#8226;s','Heat capacity at constant pressure, J/K/mol'],          'Specific density, kg/m<SUP>3</SUP>')
X_train, X_test, y_train, y_test = data_split(X,Y)
X_trained,X_tested= normalize_data(X_train, X_test)
y_trained,y_tested= log_output(y_train,y_test)
np.save('density_training_features.npy',X_trained)
np.save('density_testing_features.npy',X_tested)
np.save('density_training_target.npy',y_trained)
np.save('density_testing_target.npy',y_tested)


# X,Y=load_data('salts+descriptors+all.csv',['salt_name','name-cation','name-anion','Unnamed: 0.1','Unnamed: 0','smiles-cation','smiles-anion','Specific density, kg/m<SUP>3</SUP>','Heat capacity at constant pressure, J/K/mol'],\
#           'Viscosity, Pa&#8226;s')
# X_train, X_test, y_train, y_test = data_split(X,Y)
# X_trained,X_tested= normalize_data(X_train, X_test)
# y_trained,y_tested= log_output(y_train,y_test)
# np.save('viscosity_training_features.npy',X_trained)
# np.save('viscosity_testing_features.npy',X_tested)
# np.save('viscosity_training_target.npy',y_trained)
# np.save('viscosity_testing_target.npy',y_tested)

# X,Y=load_data('salts+descriptors+all.csv',['salt_name','name-cation','name-anion','Unnamed: 0.1','Unnamed: 0','smiles-cation','smiles-anion'],['Heat capacity at constant pressure, J/K/mol','Specific density, kg/m<SUP>3</SUP>','Viscosity, Pa&#8226;s'])
# X_train, X_test, y_train, y_test = data_split(X,Y)
# X_trained,X_tested= normalize_data(X_train, X_test)
# y_trained,y_tested= log_output(y_train,y_test)
# np.save('all_training_features.npy',X_trained)
# np.save('all_testing_features.npy',X_tested)
# np.save('all_training_target.npy',y_trained)
# np.save('all_testing_target.npy',y_tested)

# In[ ]:



