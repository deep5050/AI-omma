#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# # getting the test and train data from file

# In[234]:


train_data = pd.read_csv('./data/train.csv')
train_data= train_data.dropna(axis=0,how='any')
train_data['x'] = train_data['x']
train_data['y'] = train_data['y']
train_data['Intercept']=1 # adding an extra intercept column


# In[233]:


x_train = train_data.loc[:,['Intercept','x']]  
print(type(x_train))
print(x_train.head(10))
y_train = train_data.loc[:,'y']
print(type(y_train))
print(y_train.head(10))


# # plotting the data

# In[207]:


plt.scatter(x_train.loc[:,'x'],y_train[:],c='r',s=1,marker="x")
# plt.scatter(x_train.iloc[:,1],y_train,c="red")
plt.title("Relationship between x and y", loc="center" , size=1)
plt.xlabel("x", size=10)
plt.ylabel("y",size=10)
plt.show()


# # Implementing OLS method \hat{\beta}=\left(X^{T} X\right)^{-1} X^{T} y

# In[98]:


def linear_regression(x,y):
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
    return beta


# In[214]:


# # print(x_train.to_numpy())
# print(type(x_train.to_numpy()))
# # print(y_train.values)
# print(type(y_train.values))
beta = linear_regression(x_train.values,y_train.values)
# beta
# print(type(beta))
intercept = beta[0]
coefficient = beta[1]
print("Intercept value {}".format(intercept))
print("Coefficient value {}".format(coefficient))


# In[235]:


plt.plot(x_train.iloc[:,1],intercept+coefficient*x_train.iloc[:,1],'r--')
plt.scatter(x_train.loc[:,'x'],y_train)
plt.show()


# In[236]:


def predict(x_val):
    y_pred = intercept+coefficient*x_val
    plt.plot(x_val,y_pred,'g^')
    plt.title("Prediction for given value")
    plt.xlabel("x",size=12)
    plt.ylabel("y",size=12)
    plt.show()
    return y_pred

print(predict(67))
    


# In[ ]:




