def aard(y,y_pred):
    y = np.array(y,dtype=np.float)
    y_pred = np.array(y_pred,dtype=np.float)
    dev=abs((y-y_pred)/(y))*100
    return np.mean(dev)
