#!/usr/bin/env python
# coding: utf-8

# # Capstone project on Time Series Analysis

# ## Importing the libraries

# In[ ]:


import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from scipy.stats import boxcox

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


# ## Reading the dataset

# In[ ]:


# stores = pd.read_csv("Dataset/store.csv")
# train = pd.read_csv("Dataset/train.csv", parse_dates=True, index_col="Date", low_memory = False)

stores = pd.read_csv("../input/rossman-store-capstone/store.csv")
train = pd.read_csv("../input/rossman-store-capstone/train.csv", parse_dates=True, index_col="Date", low_memory = False)

print("Stores dataset: {}".format(stores.shape))
print("Train dataset: {}".format(train.shape))


# In[ ]:


stores.info()


# In[ ]:


train.info()


# In[ ]:


# Store is key column and is unique for all the records. Also we have a total of 1115 stores in total

stores["Store"].nunique()


# In[ ]:


# As expected in the train file, we have the time series data for the 1115 stores in 'stores' dataset

train['Store'].nunique()


# In[ ]:


# Checking the index of train dataset

train.index


# ## Checking the need to merge two dataframes

# In[ ]:


# Merging 'stores' and 'train' dataset based on the key 'Stores'

data = pd.merge(train, stores, how="inner", on="Store", validate="m:1")


# In[ ]:


print("Merged dataset shape: {}".format(data.shape))
data.info()


# In[ ]:


cols = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

for col in cols:
    print(data.loc[data['Store']==1,col].value_counts(dropna=False))
    print("-"*50)


# <b>Observations:</b><br>
# - From the above result, it is clear that if we are building individual models for each of the store, the data coming from 'stores' dataset will have same values for all the records of a store. i.e. there is no variance for the fields from 'stores dataset involved in such a case.
# - Also the promo details are provided for each date with the field 'Promo' in the train dataset and hence the 'PromoInterval' from 'stores' dataset would be redundant.

# With the above observation we proceed wihtout using any of the fields from the store dataset.
# Also since we are interested in the sales of stores [1, 3, 8, 9, 13, 25, 29, 31, 46], removing the remaining records from both 'stores' and 'train' dataset

# In[ ]:


store_ids = [1, 3, 8, 9, 13, 25, 29, 31, 46]
train = train.loc[train['Store'].isin(store_ids)]
stores = stores.loc[stores['Store'].isin(store_ids)]


# ## Analysing 'store' dataset

# In[ ]:


stores.info()


# In[ ]:


# Number of stores under consideration

stores["Store"].nunique()


# In[ ]:


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


# ## Analysing 'train' dataset

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


# In[ ]:


# Checking the value of each of the categorical fields

cols = ['DayOfWeek', 'Open', 'StateHoliday', 'SchoolHoliday', 'Promo']

for col in cols:
    print("-"*30)
    print(train[col].value_counts(dropna=False))


# In[ ]:


# For the field 'StateHoliday' we have both '0' and 0 as a value. Converting that to 0
# Also converting all the other values to 'StateHoliday' to 1 just to indicate that it was a state holiday and not the type of holiday 

train.loc[:,'StateHoliday'] = train['StateHoliday'].map({'a': 1, 'b': 1, 'c': 1, '0': 0, 0: 0}).astype('int64')
train['StateHoliday'].value_counts()


# In[ ]:


train.info()


# ### General Approach followed
# 1. Time series analysis was perfoemd on each of the stores indepedently ie by using a  data that belong to each store
# 2. All the stores under consideration were closed on Sundays and State Holidays.
# 3. We have complete dat for the stores 1, 3, 8, 9, 29, 31
# 4. Store 13 and 46 was closed nearly for half an year (from mid of 2014 to starting of 2015). Apart from that, stores were also closed on Sundays and State Holidays
# 5. Store 25 was closed for ~1 month in 2014. Thi1, 3, 8, 9, 29, 311, 3, 8, 9, 29, 31s store was also closed on Sundays and State Holidays.
# 
# #### Stores 1, 3, 8, 9, 29, 31
# 1. We have data for the 942 days for these stores.
# 2. All the stores were closed for 134 Sundays in the daterange
# 3. Few of the stores had 27 STate Holidays while the rest had 29 State Holidays.
# 4. All the stores had been closed for Sundays and State Holidays and open on all the remaining days
# 5. Sales were 0 on for those days when the store has bee closed. For the remaining days we had some data for Sales
# 6. For these stores we used linear interpolation to replace the sales values with the probable value which would be there if the store was open
# 7. Then the stationarity was checked and corresponding model was applied
# 
# #### Stores 13 and 46
# 1. We have data for the 758 days for these stores.
# 2. We didnot have any data from the mid of 2014 to starting of 2015
# 3. Two approaches were tried out. 
#  - One by just interpolating the last present values in 2014 to the value in 2015
#  - Other by resampling the values to form a continous range and then using the reindexed data for predictions
# 4. All the stores were closed for all Sundays and StateHolidays in the daterange for which we had the values for
# 5. Sales were 0 on for those days when the store has bee closed. For the remaining days we had some data for Sales
# 6. For days for which we had the store closed, we interploated the data as used for the previous set of stores
# 7. Approach 2 provided better results
# 
# #### Stores 25
# 1. We have data for the 942 days for these stores.
# 2. For date starting from 15 Jan, 2014 to 13 Feb, 2014 though the store was closed and hence the sales was 0
# 3. Since the store was closed, we had used the similar technique of interpolating the values for those records
# 4. Apart from that we have 0 sales for two dates on which the the store was open. Those two 0 records were retained
# 5. Once that is done the same approach of the previous stores were followed
# 
# 

# ### Store 1

# #### Getting the subset of 'train' dataset for store 1

# In[ ]:


train_store_1 = train.loc[train['Store']==1,:].sort_index(ascending=True)


# In[ ]:


train_store_1.info()


# In[ ]:


train_store_1.index


# #### EDA on store 1 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_1[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_1.loc[train_store_1['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_1.loc[train_store_1['DayOfWeek']==7,'Open'].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_1.loc[train_store_1['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_1.loc[train_store_1['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_1.loc[train_store_1['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_1.loc[train_store_1['Sales']==0,['Open','StateHoliday']].value_counts())


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 161 days
# - Of the 161 days, 134 were Sundays and the remaining 24 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_1.drop(columns=['Store'], inplace=True)
train_store_1.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_1.loc[:, 'Sales'] = train_store_1.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_1.loc[:, 'Customers'] = train_store_1.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_1.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_1.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_1['Sales'], whis=1.5)


# In[ ]:


fig = train_store_1['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


train_store_1['Sales'].quantile(0.95)


# In[ ]:


train_store_1['Sales'].quantile(0.99)


# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_1.loc[train_store_1['Sales'] > train_store_1['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_1['Sales'].quantile(0.97))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_1['Sales'], whis=1.5)


# In[ ]:


# from scipy.stats import boxcox
# data_boxcox = pd.Series(boxcox(train_store_1['Sales']), index=train_store_1.index)
# data_boxcox.plot(figsize=(96, 16))
# plt.legend(loc='best')
# plt.title('Sales data for Store 1')
# plt.show(block=False)


# #### Plotting the sales data

# In[ ]:


train_store_1.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 1')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_1['Sales'], model='additive')
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_1 = pd.concat([train_store_1, pd.get_dummies(train_store_1['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_1.drop(columns=['DayOfWeek'], inplace=True)
train_store_1.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_1.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - No considerable relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_1.columns


for col in cols:
    adf_test = adfuller(train_store_1[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_1['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_1['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_1['Sales'] = train_store_1['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_1.loc[:,['Sales', 'Customers']], exog=train_store_1[exog], order=(5,5), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[ ]:


train_len = 900
train_data = train_store_1[0:train_len] 
test_data = train_store_1[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index, exog=[''])


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

model_comparison=pd.DataFrame()
model_comparison.loc[0,'Store']= 1
model_comparison.loc[0,'Model']='VAR'
model_comparison.loc[0,'Variable']='Sales'
model_comparison.loc[0,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==1, 'MAPE'] = mape
model_comparison


# ### Store 3

# #### Getting the subset of 'train' dataset for store 3

# In[ ]:


train_store_3 = train.loc[train['Store']==3,:].sort_index(ascending=True)


# In[ ]:


train_store_3.info()


# In[ ]:


train_store_3.index


# #### EDA on store 3 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_3[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_3.loc[train_store_3['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_3.loc[train_store_3['DayOfWeek']==7,['Open', 'Sales']].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_3.loc[train_store_3['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_3.loc[train_store_3['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_3.loc[train_store_3['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_3.loc[train_store_3['Sales']==0,['Open','StateHoliday']].value_counts())


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 163 days
# - Of the 161 days, 134 were Sundays and the remaining 29 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_3.drop(columns=['Store'], inplace=True)
train_store_3.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_3.loc[:, 'Sales'] = train_store_3.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_3.loc[:, 'Customers'] = train_store_3.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_3.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_3.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_3['Sales'], whis=1.5)


# In[ ]:


fig = train_store_3['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


print(train_store_3['Sales'].quantile(0.95))
print(train_store_3['Sales'].quantile(0.99))


# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_3.loc[train_store_3['Sales'] > train_store_3['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_3['Sales'].quantile(0.97))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_3['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_3.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 3')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_3['Sales'], model='additive')
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_3 = pd.concat([train_store_3, pd.get_dummies(train_store_3['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_3.drop(columns=['DayOfWeek'], inplace=True)
train_store_3.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_3.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - There is a relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_3.columns


for col in cols:
    adf_test = adfuller(train_store_3[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))
    


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_3['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_3['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_3['Sales'] = train_store_3['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_3.loc[:,['Sales', 'Customers']], exog=train_store_3[exog], order=(10,4), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `Customers`');


# In[ ]:


train_len = 900
train_data = train_store_3[0:train_len] 
test_data = train_store_3[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[1,'Store']= 3
model_comparison.loc[1,'Model']='VAR'
model_comparison.loc[1,'Variable']='Sales'
model_comparison.loc[1,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==3, 'MAPE'] = mape
model_comparison


# ### Store 8

# #### Getting the subset of 'train' dataset for store 8

# In[ ]:


train_store_8 = train.loc[train['Store']==8,:].sort_index(ascending=True)


# In[ ]:


train_store_8.info()


# In[ ]:


train_store_8.index


# #### EDA on store 8 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_8[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_8.loc[train_store_8['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_8.loc[train_store_8['DayOfWeek']==7,['Open', 'Sales']].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_8.loc[train_store_8['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_8.loc[train_store_8['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_8.loc[train_store_8['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_8.loc[train_store_8['Sales']==0,['Open','StateHoliday']].value_counts())


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 161 days
# - Of the 161 days, 134 were Sundays and the remaining 24 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_8.drop(columns=['Store'], inplace=True)
train_store_8.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_8.loc[:, 'Sales'] = train_store_8.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_8.loc[:, 'Customers'] = train_store_8.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_8.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_8.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_8['Sales'], whis=1.5)


# In[ ]:


fig = train_store_8['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


print(train_store_8['Sales'].quantile(0.95))
print(train_store_8['Sales'].quantile(0.99))


# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_8.loc[train_store_8['Sales'] > train_store_8['Sales'].quantile(0.99), 'Sales'] = math.ceil(train_store_8['Sales'].quantile(0.99))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_8['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_8.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 8')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_8['Sales'], model='additive')
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_8 = pd.concat([train_store_8, pd.get_dummies(train_store_8['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_8.drop(columns=['DayOfWeek'], inplace=True)
train_store_8.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_8.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - There is a relationship between the Promo and the Sale/Customers
# - Considerable relation between Day 6 (Saturday) and the Sales/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_8.columns


for col in cols:
    adf_test = adfuller(train_store_8[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_8['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_8['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_8['Sales'] = train_store_8['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_8.loc[:,['Sales', 'Customers']], exog=train_store_8[exog], order=(20,4), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `Customers`');


# In[ ]:


train_len = 900
train_data = train_store_8[0:train_len] 
test_data = train_store_8[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[2,'Store']= 8
model_comparison.loc[2,'Model']='VAR'
model_comparison.loc[2,'Variable']='Sales'
model_comparison.loc[2,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==8, 'MAPE'] = mape
model_comparison


# ### Store 9

# #### Getting the subset of 'train' dataset for store 9

# In[ ]:


train_store_9 = train.loc[train['Store']==9,:].sort_index(ascending=True)


# In[ ]:


train_store_9.info()


# In[ ]:


train_store_9.index


# #### EDA on store 9 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_9[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_9.loc[train_store_9['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_9.loc[train_store_9['DayOfWeek']==7,['Open', 'Sales']].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_9.loc[train_store_9['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_9.loc[train_store_9['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_9.loc[train_store_9['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_9.loc[train_store_9['Sales']==0,['Open','StateHoliday']].value_counts())


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 161 days
# - Of the 161 days, 134 were Sundays and the remaining 29 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_9.drop(columns=['Store'], inplace=True)
train_store_9.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_9.loc[:, 'Sales'] = train_store_9.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_9.loc[:, 'Customers'] = train_store_9.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_9.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_9.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_9['Sales'], whis=1.5)


# In[ ]:


fig = train_store_9['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


print(train_store_9['Sales'].quantile(0.95))
print(train_store_9['Sales'].quantile(0.99))


# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_9.loc[train_store_9['Sales'] > train_store_9['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_9['Sales'].quantile(0.97))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_9['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_9.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 9')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_9['Sales'], model='additive')
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_9 = pd.concat([train_store_9, pd.get_dummies(train_store_9['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_9.drop(columns=['DayOfWeek'], inplace=True)
train_store_9.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_9.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - There is a relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_9.columns


for col in cols:
    adf_test = adfuller(train_store_9[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))
    


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_9['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_9['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_9['Sales'] = train_store_9['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_9.loc[:,['Sales', 'Customers']], exog=train_store_9[exog], order=(16,4), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `Customers`');


# In[ ]:


train_len = 900
train_data = train_store_9[0:train_len] 
test_data = train_store_9[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[3,'Store']= 9
model_comparison.loc[3,'Model']='VAR'
model_comparison.loc[3,'Variable']='Sales'
model_comparison.loc[3,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==9, 'MAPE'] = mape
model_comparison


# ### Store 29

# #### Getting the subset of 'train' dataset for store 29

# In[ ]:


train_store_29 = train.loc[train['Store']==29,:].sort_index(ascending=True)


# In[ ]:


train_store_29.info()


# In[ ]:


train_store_29.index


# #### EDA on store 29 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_29[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_29.loc[train_store_29['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_29.loc[train_store_29['DayOfWeek']==7,['Open', 'Sales']].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_29.loc[train_store_29['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_29.loc[train_store_29['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_29.loc[train_store_29['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_29.loc[train_store_29['Sales']==0,['Open','StateHoliday']].value_counts())


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 161 days
# - Of the 161 days, 134 were Sundays and the remaining 29 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_29.drop(columns=['Store'], inplace=True)
train_store_29.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_29.loc[:, 'Sales'] = train_store_29.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_29.loc[:, 'Customers'] = train_store_29.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_29.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_29.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_29['Sales'], whis=1.5)


# In[ ]:


fig = train_store_29['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


print(train_store_29['Sales'].quantile(0.95))
print(train_store_29['Sales'].quantile(0.99))


# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_29.loc[train_store_29['Sales'] > train_store_29['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_29['Sales'].quantile(0.97))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_29['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_29.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 29')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_29['Sales'], model='additive')
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_29 = pd.concat([train_store_29, pd.get_dummies(train_store_29['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_29.drop(columns=['DayOfWeek'], inplace=True)
train_store_29.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_29.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - There is a relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_29.columns


for col in cols:
    adf_test = adfuller(train_store_29[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))
    


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_29['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_29['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_29['Sales'] = train_store_29['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_29.loc[:,['Sales', 'Customers']], exog=train_store_29[exog], order=(16,4), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `Customers`');


# In[ ]:


train_len = 900
train_data = train_store_29[0:train_len] 
test_data = train_store_29[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[4,'Store']= 29
model_comparison.loc[4,'Model']='VAR'
model_comparison.loc[4,'Variable']='Sales'
model_comparison.loc[4,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==29, 'MAPE'] = mape
model_comparison


# ### Store 31

# #### Getting the subset of 'train' dataset for store 31

# In[ ]:


train_store_31 = train.loc[train['Store']==31,:].sort_index(ascending=True)


# In[ ]:


train_store_31.info()


# In[ ]:


train_store_31.index


# #### EDA on store 31 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_31[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_31.loc[train_store_31['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_31.loc[train_store_31['DayOfWeek']==7,'Open'].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_31.loc[train_store_31['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_31.loc[train_store_31['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_31.loc[train_store_31['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_31.loc[train_store_31['Sales']==0,['Open','StateHoliday']].value_counts())


# In[ ]:


print("Checking the dates where the store was open but the sales was 0")
print(train_store_31.loc[(train_store_31['Sales']==0)&(train_store_31['Open']==1),:].value_counts())


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 161 days
# - Of the 161 days, 134 were Sundays and the remaining 24 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_31.drop(columns=['Store'], inplace=True)
train_store_31.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_31.loc[:, 'Sales'] = train_store_31.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_31.loc[:, 'Customers'] = train_store_31.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_31.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_31.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_31['Sales'], whis=1.5)


# In[ ]:


fig = train_store_31['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


train_store_31['Sales'].quantile(0.95)


# In[ ]:


train_store_31['Sales'].quantile(0.99)


# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_31.loc[train_store_31['Sales'] > train_store_31['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_31['Sales'].quantile(0.97))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_31['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_31.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 31')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_31['Sales'], model='additive')
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_31 = pd.concat([train_store_31, pd.get_dummies(train_store_31['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_31.drop(columns=['DayOfWeek'], inplace=True)
train_store_31.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_31.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - No considerable relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_31.columns


for col in cols:
    adf_test = adfuller(train_store_1[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_31['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_31['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_31['Sales'] = train_store_31['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_31.loc[:,['Sales', 'Customers']], exog=train_store_1[exog], order=(5,5), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[ ]:


train_len = 900
train_data = train_store_31[0:train_len] 
test_data = train_store_31[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index, exog=[''])


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[5,'Store']= 31
model_comparison.loc[5,'Model']='VAR'
model_comparison.loc[5,'Variable']='Sales'
model_comparison.loc[5,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==31, 'MAPE'] = mape
model_comparison


# ### Store 13 - Without resampling with linear interpoloation

# In[ ]:


train_store_13 = train.loc[train['Store']==13,:].sort_index(ascending=True)


# In[ ]:


train_store_13.info()


# In[ ]:


train_store_13.index


# In[ ]:


train_store_13.isnull().sum()


# In[ ]:


train_store_13 = train_store_13.reindex(pd.date_range('2013-01-01', '2015-07-31'), fill_value=0)
train_store_13.shape


# In[ ]:


train_store_13.info()


# In[ ]:


train_store_13.isnull().sum()


# In[ ]:


train_store_13.loc[train_store_13['DayOfWeek']==0, 'DayOfWeek'] = train_store_13.loc[train_store_13['DayOfWeek']==0, :].index.to_series().dt.dayofweek + 1


# #### EDA on store 13 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_13[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_13.loc[train_store_13['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_13.loc[train_store_13['DayOfWeek']==7,'Open'].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_13.loc[train_store_13['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_13.loc[train_store_13['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_13.loc[train_store_13['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_13.loc[train_store_13['Sales']==0,['Open','StateHoliday']].value_counts())


# In[ ]:


print("Checking the dates where the store was open but the sales was 0")
print(train_store_13.loc[(train_store_13['Sales']==0)&(train_store_13['Open']==1),:].value_counts())


# In[ ]:


train_store_13['DayOfWeek'].value_counts()


# <b>Observations</b><br>
# There is a gap in the time series data from mid of 2014 to starting of 2015. Considering only those dates for which we have actual sales data, <br>
# We have 0 Sales only when the shop was closed (due to a StateHoliday or being Sunday.)<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 161 days
# - Of the 161 days, 134 were Sundays and the remaining 24 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_13.drop(columns=['Store'], inplace=True)
train_store_13.info()


# Using Linear Interpolation to fill the missing value fields for which sales were 0

# In[ ]:


train_store_13.loc[:, 'Sales'] = train_store_13.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_13.loc[:, 'Customers'] = train_store_13.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_13.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_13.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_13['Sales'], whis=1.5)


# In[ ]:


fig = train_store_13['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_13.loc[train_store_13['Sales'] > train_store_13['Sales'].quantile(0.99), 'Sales'] = math.ceil(train_store_13['Sales'].quantile(0.99))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_13['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_13.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 31')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_13['Sales'], model='additive', freq=30)
fig = decomposition.plot()
plt.show()


# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_13 = pd.concat([train_store_13, pd.get_dummies(train_store_13['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_13.drop(columns=['DayOfWeek'], inplace=True)
train_store_13.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_13.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - No considerable relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_13.columns


for col in cols:
    adf_test = adfuller(train_store_13[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_13['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_13['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_13['Sales'] = train_store_13['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_13.loc[:,['Sales', 'Customers']], exog=train_store_13[exog], order=(6,5), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[ ]:


train_len = 900
train_data = train_store_13[0:train_len] 
test_data = train_store_13[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[6,'Store']= 13
model_comparison.loc[6,'Model']='VAR - without resampling'
model_comparison.loc[6,'Variable']='Sales'
model_comparison.loc[6,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[(model_comparison['Store']==13)&(model_comparison['Model']=='VAR - without resampling'), 'MAPE'] = mape
# model_comparison.loc[model_comparison['Variable']=='Sales', 'MAPE'] = mape
model_comparison


# ### Store 13 - With resampling and linear interpoloation

# In[ ]:


train_store_13 = train.loc[train['Store']==13,:].sort_index(ascending=True)
train_store_13.reset_index(drop=True, inplace=True)


# In[ ]:


train_store_13.info()


# In[ ]:


train_store_13.isnull().sum()


# In[ ]:


train_store_13 = train_store_13.set_index(pd.date_range(start='2013-01-01', periods = len(train_store_13), freq='D'))
train_store_13.shape


# In[ ]:


train_store_13.info()


# In[ ]:


train_store_13.head()


# #### EDA on store 13 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_13[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_13.loc[train_store_13['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_13.loc[train_store_13['DayOfWeek']==7,'Open'].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_13.loc[train_store_13['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_13.loc[train_store_13['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_13.loc[train_store_13['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_13.loc[train_store_13['Sales']==0,['Open','StateHoliday']].value_counts())


# In[ ]:


print("Checking the dates where the store was open but the sales was 0")
print(train_store_13.loc[(train_store_13['Sales']==0)&(train_store_13['Open']==1),:].value_counts())


# In[ ]:


train_store_13['DayOfWeek'].value_counts()


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 137 days
# - Of the 137 days, 108 were Sundays and the remaining 29 were State Holidays
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_13.drop(columns=['Store'], inplace=True)
train_store_13.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_13.loc[:, 'Sales'] = train_store_13.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_13.loc[:, 'Customers'] = train_store_13.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# # Removing the first records since its is having 0 Sales

train_store_13.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_13.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_13['Sales'], whis=1.5)


# In[ ]:


fig = train_store_13['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_13.loc[train_store_13['Sales'] > train_store_13['Sales'].quantile(0.99), 'Sales'] = math.ceil(train_store_13['Sales'].quantile(0.99))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_13['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_13.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 31')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_13['Sales'], model='additive', freq=30)
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_13 = pd.concat([train_store_13, pd.get_dummies(train_store_13['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_13.drop(columns=['DayOfWeek'], inplace=True)
train_store_13.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_13.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - No considerable relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_13.columns


for col in cols:
    adf_test = adfuller(train_store_13[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_13['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_13['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_13['Sales'] = train_store_13['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_13.loc[:,['Sales', 'Customers']], exog=train_store_13[exog], order=(7,4), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[ ]:


train_len = 716
train_data = train_store_13[0:train_len] 
test_data = train_store_13[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[7,'Store']= 13
model_comparison.loc[7,'Model']='VAR - with resampling'
model_comparison.loc[7,'Variable']='Sales'
model_comparison.loc[7,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[(model_comparison['Store']==13)&(model_comparison['Model']=='VAR - with resampling'), 'MAPE'] = mape
# model_comparison.loc[model_comparison['Variable']=='Sales', 'MAPE'] = mape
model_comparison


# # Store 46

# In[ ]:


train_store_46 = train.loc[train['Store']==46,:].sort_index(ascending=True)
train_store_46.reset_index(drop=True, inplace=True)


# In[ ]:


train_store_46.info()


# In[ ]:


train_store_46.isnull().sum()


# In[ ]:


train_store_46 = train_store_46.set_index(pd.date_range(start='2013-01-01', periods = len(train_store_46), freq='D'))
train_store_46.shape


# In[ ]:


train_store_46.info()


# In[ ]:


train_store_46.isnull().sum()


# In[ ]:


train_store_46.head()


# #### EDA on store 46 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_46[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_46.loc[train_store_46['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_46.loc[train_store_46['DayOfWeek']==7,'Open'].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_46.loc[train_store_46['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_46.loc[train_store_46['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_46.loc[train_store_46['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_46.loc[train_store_46['Sales']==0,['Open','StateHoliday']].value_counts())


# In[ ]:


print("Checking the dates where the store was open but the sales was 0")
print(train_store_46.loc[(train_store_46['Sales']==0)&(train_store_46['Open']==1),:].value_counts())


# In[ ]:


train_store_46['DayOfWeek'].value_counts()


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.<br>
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 136 days
# - Of the 136 days, 107 were Sundays and the remaining 29 were State Holidays. There was one State Holiday which was on Sunday
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_46.drop(columns=['Store'], inplace=True)
train_store_46.info()


# Since the 'Sales' and 'Customers' are 0 only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation.
# Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open.
# But in real sense, 'Sales' would be 0 if the Store is closed

# In[ ]:


train_store_46.loc[:, 'Sales'] = train_store_46.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_46.loc[:, 'Customers'] = train_store_46.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# # Removing the first records since its is having 0 Sales

train_store_46.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_46.shape


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_46['Sales'], whis=1.5)


# In[ ]:


fig = train_store_46['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_46.loc[train_store_46['Sales'] > train_store_46['Sales'].quantile(0.99), 'Sales'] = math.ceil(train_store_46['Sales'].quantile(0.99))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_46['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_46.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 31')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_46['Sales'], model='additive', freq=30)
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_46 = pd.concat([train_store_46, pd.get_dummies(train_store_46['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_46.drop(columns=['DayOfWeek'], inplace=True)
train_store_46.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_46.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - No considerable relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_46.columns


for col in cols:
    adf_test = adfuller(train_store_46[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_46['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_46['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_46['Sales'] = train_store_46['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_46.loc[:,['Sales', 'Customers']], exog=train_store_46[exog], order=(3,4), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[ ]:


train_len = 716
train_data = train_store_46[0:train_len] 
test_data = train_store_46[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[8,'Store']= 46
model_comparison.loc[8,'Model']='VAR'
model_comparison.loc[8,'Variable']='Sales'
model_comparison.loc[8,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==46, 'MAPE'] = mape
model_comparison


# # Store 25

# In[ ]:


train_store_25 = train.loc[train['Store']==25,:].sort_index(ascending=True)
# train_store_25.reset_index(drop=True, inplace=True)


# In[ ]:


train_store_25.info()


# In[ ]:


train_store_25.isnull().sum()


# In[ ]:


train_store_25.head()


# #### EDA on store 25 dataset

# In[ ]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_25[col].value_counts())
    print("-"*40)


# In[ ]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_25.loc[train_store_25['StateHoliday']==1,['Sales','Open']].value_counts())


# In[ ]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_25.loc[train_store_25['DayOfWeek']==7,'Open'].value_counts())


# In[ ]:


print("Checking the days on which the store was closed")
print(train_store_25.loc[train_store_25['Open']==0,['Sales','DayOfWeek']].value_counts())


# In[ ]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_25.loc[train_store_25['Open']==0,['Sales','StateHoliday']].value_counts())


# In[ ]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_25.loc[train_store_25['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


# In[ ]:


print("Checking if we have 0 Sales even when the shop was open and if it was a StateHoliday or not")
print(train_store_25.loc[train_store_25['Sales']==0,['Open','StateHoliday']].value_counts())


# In[ ]:


print("Checking the dates where the store was open but the sales was 0")
print(train_store_25.loc[(train_store_25['Sales']==0)&(train_store_25['Open']==1),:])


# <b>Observations</b><br>
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.)<br>
# Also we have 0 sales for two days when the store was open
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# - Store was closed for 190 days
# - Of the 190 days, 134 were Sundays and the 29 were State Holidays. Shop was closed for the other 27 days as well (From Jan 15, 2014 to Feb 13, 2014)
# - Apart from above 190 days, we had 2 dates for which the Sales was 0 even though the shop was open
# - Store was closed for all the State Holidays and on all Sundays<br>
# 
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[ ]:


# Removing the 'Store' field since it invariant

train_store_25.drop(columns=['Store'], inplace=True)
train_store_25.info()


# We are keeping the 0 records as it is for the days with Sales 0 since its not due to any missing data. Instead of missing dates, we have records for those dates with Sales as 0

# In[ ]:


train_store_25.loc[:, 'Sales'] = train_store_25.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
train_store_25.loc[:, 'Customers'] = train_store_25.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# # Removing the first records since its is having 0 Sales

train_store_25.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_25.shape


# In[ ]:


# Chaging the 0 sales records when the shop was open back to 0

train_store_25.loc['2014-02-12','Sales'] = 0
train_store_25.loc['2014-02-12','Customers'] = 0
train_store_25.loc['2014-02-13','Sales'] = 0
train_store_25.loc['2014-02-13','Customers'] = 0


# In[ ]:


train_store_25.loc[train_store_25['Sales']==0, :]


# #### Outlier treatment

# In[ ]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_25['Sales'], whis=1.5)


# In[ ]:


fig = train_store_25['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely.
# Removing the outlier is better suited when we think the outlier is erroneous

# In[ ]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_25.loc[train_store_25['Sales'] > train_store_25['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_25['Sales'].quantile(0.97))


# In[ ]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_13['Sales'], whis=1.5)


# #### Plotting the sales data

# In[ ]:


train_store_25.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 31')
plt.show(block=False)


# #### Decomposing the time series to trend and seasonality

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_25['Sales'], model='additive', freq=30)
fig = decomposition.plot()
plt.show()


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.<br>
# Following analysis is done to find out the possible reasons for the above observations

# #### Creating dummy variables for the field 'DayOfWeek'

# In[ ]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_25 = pd.concat([train_store_25, pd.get_dummies(train_store_25['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# Removing the parent field 'DayOfWeek'
train_store_25.drop(columns=['DayOfWeek'], inplace=True)
train_store_25.info()


# #### Checking the corelation of variables

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_25.corr(), annot=True)
plt.show()


# <b>Observations</b><br>
# Following observation can be made from the above correlation plot
# - Sales and Customers fields are highly positively correlated
# - Open field is highely negatively correlated to the DayOfWeek_7 (Sunday)
# - No considerable relationship between the Promo and the Sale/Customers
# - As discovered from EDA, there is a slight correlation between StateHoliday and Open<br>
# 
# Above observation make sense from a business point of view as well.

# #### Checking the stationarity of the data

# In[ ]:


cols = train_store_25.columns


for col in cols:
    adf_test = adfuller(train_store_25[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# #### ACF and PACF plots

# In[ ]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_25['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_25['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# #### Model Creation

# In[ ]:


# preparation: input should be float type
train_store_25['Sales'] = train_store_25['Sales'] * 1.0


# In[ ]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_25.loc[:,['Sales', 'Customers']], exog=train_store_25[exog], order=(11,5), trend='n')
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[ ]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[ ]:


train_len = 900
train_data = train_store_25[0:train_len] 
test_data = train_store_25[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[ ]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index)


# In[ ]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[ ]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['Sales'], predictions['Sales'])).round(2)
#print('Income: RMSE:',  rmse)

# model_comparison=pd.DataFrame()
model_comparison.loc[9,'Store']= 25
model_comparison.loc[9,'Model']='VAR'
model_comparison.loc[9,'Variable']='Sales'
model_comparison.loc[9,'RMSE']=rmse

# Mean Absolute Percentage Error
abs_error = np.abs(test_data['Sales']-predictions['Sales'])
actual = test_data['Sales']
mape = np.round(np.mean(abs_error/actual)*100, 2)

model_comparison.loc[model_comparison['Store']==25, 'MAPE'] = mape
model_comparison

