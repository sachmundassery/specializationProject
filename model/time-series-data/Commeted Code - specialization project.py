#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math

import numpy as np
import pandas as pd
#----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt 
import seaborn as sns
#----------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
# The idea behind StandardScaler is that it will transform your data such that its distribution 
# will have a mean value 0 and standard deviation of 1
#----------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
#----------------------------------------------------------------------------------------------------
from scipy.stats import boxcox
# SciPy's stats package provides a function called boxcox for performing box-cox power transformation 
# that takes in original non-normal data as input and returns fitted data along with the lambda value 
# that was used to fit the non-normal distribution to normal distribution.
#----------------------------------------------------------------------------------------------------
import statsmodels.api as sm
#statsmodels is a Python module that provides classes and functions for the estimation of many different 
# statistical models, as well as for conducting statistical tests, and statistical data exploration.
#----------------------------------------------------------------------------------------------------
from statsmodels.tsa.stattools import adfuller
# Augmented Dickey Fuller test (ADF Test) is a common statistical test used to test whether a given 
# Time series is stationary or not. The adfuller function returns a tuple of statistics from the ADF 
# test such as the Test Statistic, P-Value, Number of Lags Used, Number of Observations used for the 
# ADF regression and a dictionary of Critical Values.
#----------------------------------------------------------------------------------------------------
from statsmodels.tsa.stattools import kpss
# KPSS test is a statistical test to check for stationarity of a series around a deterministic trend. 
# Like ADF test, the KPSS test is also commonly used to analyse the stationarity of a series.
#----------------------------------------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf
# A correlogram (also called Auto Correlation Function ACF Plot or Autocorrelation plot) is a visual way 
# to show serial correlation in data that changes over time (i.e. time series data). Serial correlation 
# (also called autocorrelation) is where an error at one point in time travels to a subsequent point in time.
# A plot of the autocorrelation of a time series by lag is called the AutoCorrelation Function, or the acronym ACF. 
# This plot is sometimes called a correlogram or an autocorrelation plot.Running the example creates a 2D plot 
# showing the lag value along the x-axis and the correlation on the y-axis between -1 and 1.
#----------------------------------------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_pacf


# In[ ]:


# fetching the required csv files
stores = pd.read_csv("../input/rossman-store-capstone/store.csv")
train = pd.read_csv("../input/rossman-store-capstone/train.csv", parse_dates=True, index_col="Date", low_memory = False)
#----------------------------------------------------------------------------------------------------
# to get the dimensions of the dataset
print("Stores dataset: {}".format(stores.shape))
print("Train dataset: {}".format(train.shape))


# In[ ]:


stores.info()
train.info()


# In[ ]:


# nunique() function return number of unique elements in the object. 
# Store is key column and is unique for all the records. Also we have a total of 1115 stores in total

stores["Store"].nunique()
#----------------------------------------------------------------------------------------------------
# As expected in the train file, we have the time series data for the 1115 stores in 'stores' dataset

train['Store'].nunique()
#----------------------------------------------------------------------------------------------------
# Checking the index of train dataset
# it will display the index on which the rows are identified 
train.index


# In[ ]:


# Checking the need to merge 2 data frames


# In[ ]:


# Merging 'stores' and 'train' dataset based on the key 'Stores'

data = pd.merge(train, stores, how="inner", on="Store", validate="m:1")

# paramters : 
# 1. Name of the 1st data frame object
# 2. Name of the 2nd data frame object
# 3. inner join : Use intersection of keys
# 4. on − Columns (names) to join on. Must be found in both the left and right DataFrame objects.


# In[ ]:


print("Merged dataset shape: {}".format(data.shape))
data.info()


# In[ ]:


cols = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

for col in cols:
    print(data.loc[data['Store']==1,col].value_counts(dropna=False))
    print("-"*50)

# loc[rows,columns]
# rows : selecting rows with 1 in 'store' column   


# In[ ]:


#--------------------------------------------------------------------------------------------


# In[ ]:


# these are the stores we are selecting for our study
store_ids = [1, 3, 8, 9, 13, 25, 29, 31, 46]

# now we are re-assigning our dataframe with the informations of the required stores
train = train.loc[train['Store'].isin(store_ids)]
stores = stores.loc[stores['Store'].isin(store_ids)]

# isin(): Pandas isin() method is used to filter data frames. 
# isin() method helps in selecting rows with having a particular(or Multiple) value in a particular column


# # Analysing 'store' dataset

# In[ ]:


stores.info()


# In[ ]:


# Number of stores under consideration

stores["Store"].nunique()
# ie. we are considering only 9 unique stores


# In[ ]:


# selecting some specific columns of stores dataset
cols = ['StoreType', 'Assortment', 'Promo2']

for col in cols:
    print(stores[col].value_counts())


# In[ ]:


print("Number of stores that has not run any promo: {}".format(stores.loc[stores['Promo2']==0,'Store'].count()))
print("Stores that hasn't run any promo: ")
print(stores.loc[stores['Promo2']==0,'Store'].values)


# In[ ]:


# Checking if the promo fields are NaN for stores that has 'Promo2' as 0
print(stores.loc[stores['Promo2']==0,'Promo2SinceWeek'].value_counts(dropna=False))
print(stores.loc[stores['Promo2']==0, 'Promo2SinceYear'].value_counts(dropna=False))
print(stores.loc[stores['Promo2']==0, 'PromoInterval'].value_counts(dropna=False))


# Analysing 'train' dataset

# In[ ]:


train.info()


# In[ ]:


# Checking if we have the data for all the dates for each of the store

print(train.groupby('Store').size().value_counts())

# We have data for 942 dates for 934 stores, while for 180 stores we only have data for 758 dates 
# and for 1 store we have data for 941 dates
stores_942 = train.groupby('Store').size()[train.groupby('Store').size()==942].index.to_list()
print(stores_942)

stores_758 = train.groupby('Store').size()[train.groupby('Store').size()==758].index.to_list()
print(stores_758)

# stores_941 = train.groupby('Store').size()[train.groupby('Store').size()==941].index.to_list()
# print(len(stores_941))


# In[ ]:


# # Converting the date field in to a datetime object

# train.loc[:,'Date'] = pd.to_datetime(train.loc[:,'Date'], format='%Y-%m-%d')

