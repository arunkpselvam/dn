#!/usr/bin/env python
# coding: utf-8

# # Multivariate Time Series Forecasting

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import tensorflow as tf
from tensorflow import keras

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# ## Air Pollution Forecasting
# 
# https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data
# 
# Attribute Information:
# 
#  - No    : row number
#  - year  : year of data in this row
#  - month : month of data in this row
#  - day   : day of data in this row
#  - hour  : hour of data in this row
#  - pm2.5 : PM2.5 concentration (ug/m^3)
#  - DEWP  : Dew Point 
#  - TEMP  : Temperature
#  - PRES  : Pressure (hPa)
#  - cbwd  : Combined wind direction
#  - Iws   : Cumulated wind speed (m/s)
#  - Is    : Cumulated hours of snow
#  - Ir    : Cumulated hours of rain 
#  
#  Ref: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# ## Dataset preparation

# In[2]:


# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

dataset = pd.read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], 
                      index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)

# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)

# drop the first 24 hours
dataset = dataset[24:]

# summarize first 5 rows
print(dataset.head(5))

# save to file
dataset.to_csv('pollution.csv')


# In[3]:


# load dataset
dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values

# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1

# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()


# In[4]:


# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[5]:


# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())


# ### Now you have X and y, slice them into training and test dataset.

# In[6]:


# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
Xtrain, Ytrain = train[:, :-1], train[:, -1]
Xtest, Ytest = test[:, :-1], test[:, -1]

### RNN needs 3D input
# reshape input to be 3D [samples, timesteps, features]
Xtrain = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))
Xtest = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))
print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)


# ## GRU RNN

# In[7]:


gruModel = keras.models.Sequential()

gruModel.add(keras.layers.GRU(50, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))

gruModel.add(keras.layers.Dense(1))

gruModel.summary()


# In[8]:


# Configure  the model for training, by using appropriate optimizers and regularizations
# Available optimizer: adam, rmsprop, adagrad, sgd
# loss:  objective that the model will try to minimize. 
# Available loss: categorical_crossentropy, binary_crossentropy, mean_squared_error
# metrics: List of metrics to be evaluated by the model during training and testing. 

gruModel.compile(loss='mae', optimizer='adam', metrics=['mae'])


# In[9]:


# train the model

history = gruModel.fit(Xtrain, Ytrain, epochs = 50, batch_size=72, validation_split=0.1, verbose=1 )


# In[10]:


# plotting training and validation loss

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[11]:


testResult = gruModel.evaluate(Xtest, Ytest)
print(testResult)


# # Exercise 
# 
# Modify the code as per the below instructions
# - Use a different dataset.
# - Choose a different embedding like word2vec or gensim, if applicable.
# - Modify the architecture, as below
#     - Single layer of GRU and FC
#     - Two layers of GRU and FC
# - Change the number of GRU units in each layer.
# 
