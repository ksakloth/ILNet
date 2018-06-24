
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import time
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, SpatialDropout2D, Flatten, Activation, merge, Input, Masking
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, LearningRateScheduler
from keras.regularizers import l2, l1
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
from math import sqrt
import scipy
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix,mean_squared_error,r2_score


# In[ ]:

mode='batch'


# In[ ]:

if mode == 'interactive':
    taskname = 'il'
    pathname = 'chemception'
elif mode == 'batch':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='IL')
    parser.add_argument('task')
    parser.parse_args(sys.argv[1:])


# In[ ]:

if mode == "interactive":
    archdir = "archive/archive/"
elif mode == "batch":
    archdir = "../../archive/archive/"
    
jobname='il'
taskname='descriptors'


# In[ ]:

def load_data(taskname,ion,dataset,category):
    filename=str(archdir+taskname+ion+dataset+category+".npy")
    arr = np.load(filename)
    print(arr.shape)
    return arr
X_cation_train=load_data('viscosity_','cation_','training_','features')
X_anion_train=load_data('viscosity_','anion_','training_','features')
X_TP_train=load_data('viscosity_','TP_','training_','features')

X_cation_test=load_data('viscosity_','cation_','testing_','features')
X_anion_test=load_data('viscosity_','anion_','testing_','features')
X_TP_test=load_data('viscosity_','TP_','testing_','features')

y_train=load_data('viscosity_','salt_','training_','target')
y_test=load_data('viscosity_','salt_','testing_','target')


# In[ ]:

def prep_X(array):                        #tabular data 
    X_sample=array.astype('float32')
    print(X_sample.shape)
    return X_sample


# In[ ]:

def prep_Y(array):
    y_sample = array.astype("float32")
    print(y_sample.shape)
    return y_sample


# In[ ]:

X_cation_train = prep_X(X_cation_train)
X_anion_train = prep_X(X_anion_train)
X_TP_train = prep_X(X_TP_train)


X_cation_test = prep_X(X_cation_test)
X_anion_test = prep_X(X_anion_test)
X_TP_test = prep_X(X_TP_test)

y_train = prep_Y(y_train)
y_test = prep_Y(y_test)


# In[ ]:

n_fold=5
batch_size=30
epoch=500
np.random.seed(7)


# In[ ]:

def setup_cation_mlp(params):
    mlp_input = Input(shape=(int(X_cation_train.shape[1]),))
    x = Dense(params['mlp1_units'],init='glorot_normal', activation="relu")(mlp_input)
    x = Dropout(0.5)(x)
    x = Dense(params['mlp2_units'],init='glorot_normal', activation="relu")(x)
    x = Dropout(0.5)(x)
    
    model = Model(mlp_input, x)
    print(model.summary())
    model.compile(optimizer="adam", loss="mean_squared_error",metrics=['mse'])
    return model


# In[ ]:

def setup_anion_mlp(params):
    mlp_input = Input(shape=(int(X_anion_train.shape[1]),))
    x = Dense(params['mlp1_units'],init='glorot_normal', activation="relu")(mlp_input)
    x = Dropout(0.5)(x)
    x = Dense(params['mlp2_units'],init='glorot_normal',activation="relu")(x)
    x = Dropout(0.5)(x)
    
    model = Model(mlp_input, x)
    print(model.summary())
    model.compile(optimizer="adam", loss="mean_squared_error",metrics=['mse'])
    return model


# In[ ]:

def setup_TP_mlp(params):
    mlp_input = Input(shape=(int(X_TP_train.shape[1]),))
    x = Dense(params['mlp1_units'],init='glorot_normal',activation="relu")(mlp_input)
    x = Dropout(0.5)(x)
    x = Dense(params['mlp2_units'],init='glorot_normal',activation="relu")(x)
    x = Dropout(0.5)(x)
    
    model = Model(mlp_input, x)
    print(model.summary())
    model.compile(optimizer="adam", loss="mean_squared_error",metrics=['mse'])
    return model


# In[ ]:

def setup_hybrid(model1,model2,model3,params):
   
    
    mlp_input1 = Input(shape=(int(X_cation_train.shape[1]),))
    mlp_input2 = Input(shape=(int(X_cation_train.shape[1]),))
    mlp_input3 = Input(shape=(int(X_TP_train.shape[1]),))
    
    x1=model1(mlp_input1)
    x2=model2(mlp_input2)
    #x3=model3(mlp_input3)
    
    merged_out=merge([x1,x2,mlp_input3],mode='concat') #add mlp_input3 where x3
    #dense layer here of 128ish neurons. # for interactions, maybe try 2 or 3
    x=Dense(params['hybrid1_units'], activation='relu')(merged_out)    
    x = Dropout(0.1)(x)                          
    x=Dense(params['hybrid2_units'], activation='relu')(x)                     
    x = Dropout(0.1)(x)
    x=Dense(params['hybrid3_units'], activation='relu')(x)                     
    x = Dropout(0.1)(x)
    x=Dense(1,activation='linear')(x)
    #no net for T,P and then add in merge direct
    
    final_model = Model(input=[mlp_input1,mlp_input2,mlp_input3], output=[x])
    final_model.compile(optimizer="adam", loss="mean_squared_error")
    
    print(final_model.summary()) #ALWAYS CHECK CONNECTIONS
    
    return(final_model)


# In[ ]:

def rmse(y,y_pred):
    rms=sqrt(mean_squared_error(y,y_pred))
    return rms


# In[ ]:

def aard(y,y_pred):
    y = np.array(y,dtype=np.float)
    y_pred = np.array(y_pred,dtype=np.float)
    return np.mean(abs((np.exp(y)-np.exp(y_pred))/np.exp(y)))*100


# In[ ]:

def avg(x):
    return np.mean(x)


# In[ ]:

def final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test):
    kf = KFold(n_splits=n_fold, shuffle=False, random_state=7) 
    fold=0
    y_pred_val_all=[]
    y_pred_test_all=[]
    
    y_val_all=[]
    y_test_all=[]
    
    cv_train_rmse=[]
    cv_train_r2=[]
    cv_train_aard=[]

    cv_val_rmse=[]
    cv_val_r2=[]
    cv_val_aard=[]
      
    cv_test_rmse=[]
    cv_test_r2=[]
    cv_test_aard=[]
    for train_index, test_index in kf.split(X_cation_train,X_anion_train):
        fold=fold+1
        print('fold: '+str(fold))
        X_cation_trainn, X_cation_val = X_cation_train[train_index], X_cation_train[test_index]
        X_anion_trainn, X_anion_val = X_anion_train[train_index], X_anion_train[test_index]
        X_TP_trainn, X_TP_val = X_TP_train[train_index], X_TP_train[test_index]
        y_trainn, y_val = y_train[train_index], y_train[test_index]
        
        #printing shapes
        print('Train set: '+str(X_cation_trainn.shape)+str(y_trainn.shape))
        print('Validation set: '+str(X_cation_val.shape)+str(y_val.shape))
        print('Test set: '+str(X_cation_test.shape)+str(y_test.shape))
        
        #calling models
        model1=setup_cation_mlp(params)
        model2=setup_anion_mlp(params)
        model3=setup_TP_mlp(params)
        model=setup_hybrid(model1,model2,model3,params)
        
        print(model.summary())
        
        model_json = model.to_json()
        filepath=str(taskname+"_")+str("architecture_")+str(fold)+str(".json")
        with open(filepath, "w") as json_file:
            json_file.write(model_json)
            
        early = EarlyStopping(monitor='val_loss', patience=50, verbose=1) #specify the validation data in fit #reload the best model 
        saveBestModel = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
        
        model.fit([X_cation_trainn,X_anion_trainn,X_TP_trainn],y_trainn,validation_data=([X_cation_val,X_anion_val,X_TP_val],y_val),nb_epoch=epoch,batch_size=batch_size,callbacks=[early,saveBestModel],verbose=2)
        model.load_weights(filepath)
        
        final_loss = model.evaluate([X_cation_val,X_anion_val,X_TP_val], y_val, verbose=0) #look at loss curve
        
        #load BEST model here to do this here. 
        y_pred_train=model.predict([X_cation_trainn,X_anion_trainn,X_TP_trainn])
        y_pred_val=model.predict([X_cation_val,X_anion_val,X_TP_val])
        y_pred_test=model.predict([X_cation_test,X_anion_test,X_TP_test])
        
        y_pred_val_all.append(y_pred_val)
        y_pred_test_all.append(y_pred_test)
    
        y_val_all.append(y_val)
        y_test_all.append(y_test)
        
        cv_train_rmse.append(rmse(y_trainn,y_pred_train))
        cv_train_r2.append(r2_score(y_trainn,y_pred_train))
        cv_train_aard.append(aard(y_trainn,y_pred_train))
    
        cv_val_rmse.append(rmse(y_val,y_pred_val))
        cv_val_r2.append(r2_score(y_val,y_pred_val))
        cv_val_aard.append(aard(y_val,y_pred_val))
    
        cv_test_rmse.append(rmse(y_test,y_pred_test))
        cv_test_r2.append(r2_score(y_test,y_pred_test))
        cv_test_aard.append(aard(y_test,y_pred_test))
        
    np.save('pred_val.npy',y_pred_val_all)
    np.save('pred_test.npy',y_pred_test_all)
    np.save('val.npy',y_val_all)
    np.save('test.npy',y_test_all)
    
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


# In[ ]:

#16


# In[ ]:

params= {"mlp1_units":16, "mlp2_units":16,'hybrid1_units':16,'hybrid2_units':16,'hybrid3_units':16}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":16, "mlp2_units":16,'hybrid1_units':32,'hybrid2_units':32,'hybrid3_units':32}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":16, "mlp2_units":16,'hybrid1_units':64,'hybrid2_units':64,'hybrid3_units':64}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":16, "mlp2_units":16,'hybrid1_units':128,'hybrid2_units':128,'hybrid3_units':128}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":16, "mlp2_units":16,'hybrid1_units':256,'hybrid2_units':256,'hybrid3_units':256}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

#32


# In[ ]:

params= {"mlp1_units":32, "mlp2_units":32,'hybrid1_units':16,'hybrid2_units':16,'hybrid3_units':16}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":32, "mlp2_units":32,'hybrid1_units':32,'hybrid2_units':32,'hybrid3_units':32}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":32, "mlp2_units":32,'hybrid1_units':64,'hybrid2_units':64,'hybrid3_units':64}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":32, "mlp2_units":32,'hybrid1_units':128,'hybrid2_units':128,'hybrid3_units':128}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":32, "mlp2_units":32,'hybrid1_units':256,'hybrid2_units':256,'hybrid3_units':256}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

#64


# In[ ]:

params= {"mlp1_units":64, "mlp2_units":64,'hybrid1_units':16,'hybrid2_units':16,'hybrid3_units':16}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":64, "mlp2_units":64,'hybrid1_units':32,'hybrid2_units':32,'hybrid3_units':32}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":64, "mlp2_units":64,'hybrid1_units':64,'hybrid2_units':64,'hybrid3_units':64}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":64, "mlp2_units":64,'hybrid1_units':128,'hybrid2_units':128,'hybrid3_units':128}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":64, "mlp2_units":64,'hybrid1_units':256,'hybrid2_units':256,'hybrid3_units':256}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

#128


# In[ ]:

params= {"mlp1_units":128, "mlp2_units":128,'hybrid1_units':16,'hybrid2_units':16,'hybrid3_units':16}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":128, "mlp2_units":128,'hybrid1_units':32,'hybrid2_units':32,'hybrid3_units':32}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":128, "mlp2_units":128,'hybrid1_units':64,'hybrid2_units':64,'hybrid3_units':64}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":128, "mlp2_units":128,'hybrid1_units':128,'hybrid2_units':128,'hybrid3_units':128}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":128, "mlp2_units":128,'hybrid1_units':256,'hybrid2_units':256,'hybrid3_units':256}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

#256


# In[ ]:

params= {"mlp1_units":256, "mlp2_units":256,'hybrid1_units':16,'hybrid2_units':16,'hybrid3_units':16}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":256, "mlp2_units":256,'hybrid1_units':32,'hybrid2_units':32,'hybrid3_units':32}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":256, "mlp2_units":256,'hybrid1_units':64,'hybrid2_units':64,'hybrid3_units':64}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":256, "mlp2_units":256,'hybrid1_units':128,'hybrid2_units':128,'hybrid3_units':128}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:

params= {"mlp1_units":256, "mlp2_units":256,'hybrid1_units':256,'hybrid2_units':256,'hybrid3_units':256}
print('metrics of train,val,test : '+str(params)+str(final(n_fold,X_cation_train,X_anion_train,X_TP_train,y_train,y_test)))


# In[ ]:



