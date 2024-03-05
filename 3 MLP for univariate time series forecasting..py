#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
from numpy import array

from keras.models import Sequential
from keras.layers import Dense


# ## Transform univariate time series to supervised learning problem

# In[2]:


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
    


# In[4]:


# define univariate time series
# define input sequence
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# choose a number of time steps
n_steps = 3

# transform to a supervised learning problem
X, y = split_sequence(raw_seq, n_steps)
print(X.shape, y.shape)

# show each sample
for i in range(len(X)):
    print(X[i], y[i])


# In[5]:


# transform input from [samples, features] to [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
print(X.shape)


# ## Model

# In[8]:


# MLP with an input layer

model = Sequential()
model.add(Dense(100, activation= 'relu' , input_dim=n_steps))
model.add(Dense(1))
model.summary()

# configure the model
model.compile(optimizer= 'adam' , loss= 'mse' )


# In[12]:


# Training
# fit model
history = model.fit(X, y, epochs=1000, verbose=0)


# In[13]:


# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps))

yhat = model.predict(x_input, verbose=0)
print(yhat)


# # Exercise 
# 
# Modify the code as per the below instructions
# - Use a different dataset, such that it shows both trend and seasonality. The data can be univariate or multivariate.
# - Modify the architecture, as below
#     - CNN with 2 conv layers. Rest is as per our choice. 
#     - RNN with one layer
# - Compare the results of MLP (as given), CNN and RNN.
