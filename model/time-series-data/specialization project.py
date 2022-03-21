#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:



# index_col is made as 'date'
# parse_date : to convert string 'date' to 'date' datatype

stores = pd.read_csv("store.csv")
train = pd.read_csv("train.csv", parse_dates=True, index_col="Date", low_memory = False)

print("Stores dataset: {}".format(stores.shape))
print("Train dataset: {}".format(train.shape))


# In[3]:


stores.info()


# In[4]:


train.info()


# In[5]:


# Store is key column and is unique for all the records. Also we have a total of 1115 stores in total

stores["Store"].nunique()


# In[6]:


# As expected in the train file, we have the time series data for the 1115 stores in 'stores' dataset

train['Store'].nunique()


# In[7]:


# Checking the index of train dataset

train.index
# index is date and its data type is dateTime, because we parse_date=True


# In[8]:


# Checking the need to merge 2 data frames


# In[9]:


# Merging 'stores' and 'train' dataset based on the key 'Stores'

data = pd.merge(train, stores, how="inner", on="Store", validate="m:1")

# based on 'Store' column we are merging. this column is present in both data set
# validate="m:1" - relationship between train:stores is m:1 because for 1 store in 
# 'stores dataset' we have 'm' records in 'train dataset


# In[10]:


print("Merged dataset shape: {}".format(data.shape))
data.info()


# In[11]:


cols = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
# these columns are present in 'stores dataset' 


for col in cols:
    # we are considering only store 1, taking a particualr case. this is same for all stores
    print(data.loc[data['Store']==1,col].value_counts(dropna=False))
    
    # value_counts : by column, if 10 records, sachin -3, regil-7 , would be the output
    # total count of 'c' in StoreType is 942
    print("-"*50)

    
# here number of each type is constant, ie 942


# In[12]:


store_ids = [1, 3, 8, 9, 13, 25, 29, 31, 46]
# no particular reason for selecting these stores

train = train.loc[train['Store'].isin(store_ids)]
# in the train dataset, we are selecting all rows whose store_ids of the "store" column match


stores = stores.loc[stores['Store'].isin(store_ids)]
# no much relavance


# In[13]:


#Analysing 'store' dataset ( no much relavance)


# In[14]:


stores.info()


# In[15]:


# Number of stores under consideration

stores["Store"].nunique()


# In[16]:


cols = ['StoreType', 'Assortment', 'Promo2']

for col in cols:
    print(stores[col].value_counts())


# In[17]:


print("Number of stores that has not run any promo: {}".format(stores.loc[stores['Promo2']==0,'Store'].count()))
print("Stores that hasn't run any promo: ")
print(stores.loc[stores['Promo2']==0,'Store'].values)


# since there are 6 stores with no promos we will confirm that they are actually NaN for 'Promo2SinceWeek',  'Promo2SinceYear','PromoInterval'


# In[18]:


# Checking if the promo fields are NaN for stores that has 'Promo2' as 0
print(stores.loc[stores['Promo2']==0,'Promo2SinceWeek'].value_counts(dropna=False))
print(stores.loc[stores['Promo2']==0, 'Promo2SinceYear'].value_counts(dropna=False))
print(stores.loc[stores['Promo2']==0, 'PromoInterval'].value_counts(dropna=False))


# ## Analysing 'train' dataset

# In[19]:


train.info()


# In[20]:


# Checking if we have the data on all dates for each store ie we are checking whether those 9 stores have 2.5 yrs dates

print(train.groupby('Store').size().value_counts())

# We have data for 942 dates for 7 stores, while for 2 stores we only have data for 758 dates 

stores_942 = train.groupby('Store').size()[train.groupby('Store').size()==942].index.to_list()
# "Stores with 942 dates"
print(stores_942)

stores_758 = train.groupby('Store').size()[train.groupby('Store').size()==758].index.to_list()
# "Stores with 758 dates"
print(stores_758)


# In[22]:


# Checking the value of each of the categorical fields (fields that are not continuous)

cols = ['DayOfWeek', 'Open', 'StateHoliday', 'SchoolHoliday', 'Promo']

for col in cols:
    print("-"*30)
    print(train[col].value_counts(dropna=False))


# In[23]:


# For the field 'StateHoliday' we have both '0' and 0 as a value. Converting that to 0
# Also converting all the other values to 'StateHoliday' to 1 just to indicate that it was a state holiday and not the type of holiday 

train.loc[:,'StateHoliday'] = train['StateHoliday'].map({'a': 1, 'b': 1, 'c': 1, '0': 0, 0: 0}).astype('int64')
train['StateHoliday'].value_counts()


# In[24]:


train.info()


# Store 1 

# In[25]:


train_store_1 = train.loc[train['Store']==1,:].sort_index(ascending=True) # arrange the data according to date


# In[26]:


train_store_1.info()


# In[27]:


train_store_1.index


# EDA on store 1 dataset

# In[28]:


print("Checking the number of StateHolidays, SchoolHoliday and Open days")

cols = ['StateHoliday', 'SchoolHoliday', 'Open', 'Promo']
for col in cols:
    print(train_store_1[col].value_counts())
    print("-"*40)
    
# this data is for the 942 days 


# In[29]:


print("Checking if the store was open on a StateHoliday and the corresponding Sales")
print(train_store_1.loc[train_store_1['StateHoliday']==1,['Sales','Open']].value_counts())

# 27 in the output is actually the count of total number of state holidays
# inference : on state holidays , the shop was closed (open=0) and therefore te sales = 0


# In[30]:


print("Checking if the store was open on any Sundays and the corresponding Sales")
print(train_store_1.loc[train_store_1['DayOfWeek']==7,'Open'].value_counts())

# there are 134 sundays, but it was closed


# In[31]:


print("Checking the days on which the store was closed")
print(train_store_1.loc[train_store_1['Open']==0,['Sales','DayOfWeek']].value_counts()) 

# as you can see from the output, daysOfWeek doesnt have '6', ie on all saturdays the shops was open
# 3rd column in the total number of days the shop was closed for each day in datysOfWeek


# In[32]:


print("Checking the Sales when the shop was closed and if it was a StateHoliday or not")
print(train_store_1.loc[train_store_1['Open']==0,['Sales','StateHoliday']].value_counts())

# from the output, stateholiday=1 -> shop was closed on stateholiday, ie out of 161 closed days, 134 was because of sunday
# and the rest 27 days was because of state holiday


# In[33]:


print("Checking the SchoolHolidays for which the store was open and the corresponding value for StateHoliday")
print(train_store_1.loc[train_store_1['SchoolHoliday']==1,['DayOfWeek', 'Open', 'SchoolHoliday']].value_counts())


#school holiday ayitum tuesday 30 thavana open arnu. like that everything


# In[34]:


print("Checking if we have 0 Sales even when the shop was open")
print(train_store_1.loc[train_store_1['Sales']==0,['Open','StateHoliday']].value_counts())

# kada open ayitondengil sale ond


# Observations
# We have 0 Sales only when the shop was closed (either due to a StateHoliday or being Sunday.
# From the above outputs we can make the following observations between 'Open' and 'StateHoliday' fields
# 
# Store was closed for 161 days
# Of the 161 days, 134 were Sundays and the remaining 27 were State Holidays
# Store was closed for all the State Holidays and on all Sundays
# No evident observations between the above two fields with the field 'SchoolHoliday'

# In[35]:


# Removing the 'Store' field since it invariant
# since we are working only on store 1 , the value in 'Store' column will be for 'store 1' only
# so there is no point in having that column, so remove it.

train_store_1.drop(columns=['Store'], inplace=True)
train_store_1.info()


# Since the 'Sales' and 'Customers' are 0, only when the store was closed (due to pulic or state holiday), we can consider it as a missing value and for our purpose impute with the values that could have been present if the store had been open. For that we use Linear Interpolation. Even in predction, we could use the same method, i.e comparing the predicted value with what could have been the 'Sales' if the Store was open. But in real sense, 'Sales' would be 0 if the Store is closed

# In[36]:


train_store_1.loc[:, 'Sales'] = train_store_1.loc[:, 'Sales'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')
# if x==0, we change 0 to NaN, else put what ever value is there in x.

train_store_1.loc[:, 'Customers'] = train_store_1.loc[:, 'Customers'].map(lambda x:np.NaN if x==0 else x).interpolate(method='linear')

# Removing the first records since its is having 0 Sales

train_store_1.drop(index=pd.to_datetime('2013-01-01'),axis=0, inplace=True)
train_store_1.shape


# Outlier treatment

# In[37]:


fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_1['Sales'], whis=1.5)


# In[38]:


fig = train_store_1['Sales'].hist(figsize = (12,4))


# Since from the time series plot we had observed a general increase in sales in the months of Dec-Jan, it is better to replace the higher values with lower extreme (Winsorizing) rather than removing the value completely. Removing the outlier is better suited when we think the outlier is erroneous

# In[39]:


train_store_1['Sales'].quantile(0.95)


# In[40]:


train_store_1['Sales'].quantile(0.99)


# In[41]:


# Replacing the outliers (above 99 quantile) to the value at 99 percentile

train_store_1.loc[train_store_1['Sales'] > train_store_1['Sales'].quantile(0.97), 'Sales'] = math.ceil(train_store_1['Sales'].quantile(0.97))


# In[43]:


# Boxplot after winsorizing the outliers

fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train_store_1['Sales'], whis=1.5)


# Plotting the sales data of 2.5yrs 

# In[45]:


train_store_1.loc[:,'Sales'].plot(figsize=(96, 16))
plt.legend(loc='best')
plt.title('Sales data for Store 1')
plt.show(block=False)


# Decomposing the time series to trend and seasonality

# In[47]:


from pylab import rcParams
rcParams['figure.figsize'] = (96,16)
decomposition = sm.tsa.seasonal_decompose(train_store_1['Sales'], model='additive')
fig = decomposition.plot()
plt.show()

# trend manasilakanolla graphs, athrollu


# The above plot tells us that there is a general increase in the months Dec-Jan and also there is repeating 0 sales throught out the timeframe.
# 

# Creating dummy variables for the field 'DayOfWeek' because, it has categorical values, and they are 1 ,2,...7, we need to change it. 
# like if monday is denoted as 1 , tuesday is denoted as 2.. like wise
# for each day of the weeek we will create a column..
# and on that particular row, if the it is monday..other than monday column,rest every days of the week column will be assigned as 0 . monday column would get 1..like wise

# In[49]:


# Creating dummy variables and concatenating the dummy variables of field 'DayOfWeek' to the original dataframe
train_store_1 = pd.concat([train_store_1, pd.get_dummies(train_store_1['DayOfWeek'], prefix='DayOfWeek', drop_first=True)], axis=1)

# drop_first=True : 7 days ind..but 6 daysum 0 anengi obviously 1 will be on the 7th day, it is understood. the value will
# be assigned automaticaly by the function , we need not bother about it
# pd.concat train_store1 nnu parayana dataframe inte attath aa 6 columns will be attached.


# Removing the parent field 'DayOfWeek', karanam.. namuk ini venda..namma split cheythalo
train_store_1.drop(columns=['DayOfWeek'], inplace=True)
train_store_1.info()

# now our entire dataframe has 0 or 1, except for sales(number of sales) and customers(number of customers)


# Checking the corelation of variables¶

# In[51]:


plt.figure(figsize=(12,8))
sns.heatmap(train_store_1.corr(), annot=True)
plt.show()


# Checking the stationarity of the data
# 
# our data must be stationary, only then we can apply time series analyis

# In[52]:


cols = train_store_1.columns

# adf test is done on all columns
# p values must be less than 0.05, if greater -> not stationary, thus we cannot apply time series analysis, thus we might need 
# to make it stationary

for col in cols:
    adf_test = adfuller(train_store_1[col])

    print('ADF Statistic ({}): {}'.format(col, round(adf_test[0], 4)))
    print('Critical Values @ 0.05 ({}): {}'.format(col, round(adf_test[4]['5%'], 2)))
    print('p-value ({}): {}'.format(col, round(adf_test[1], 4)))


# ACF and PACF plots,
# 
# lag : innathe sales ariyan cheyan yethra dhivasathe data vendi vannu... 1 day mathiyengi lag =1

# In[54]:


plt.figure(figsize=(18,4))
plt.subplot(1,2,1)
plot_pacf(train_store_1['Sales'] , ax=plt.gca(), lags = 50)
plt.subplot(1,2,2)
plot_acf(train_store_1['Sales'] , ax=plt.gca(), lags = 50)
plt.show()


# Model Creation

# In[55]:


# preparation: input should be float type
train_store_1['Sales'] = train_store_1['Sales'] * 1.0


# In[56]:


exog = ['Promo', 'StateHoliday', 'SchoolHoliday', 'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4', 'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7']
mod = sm.tsa.VARMAX(train_store_1.loc[:,['Sales', 'Customers']], exog=train_store_1[exog], order=(5,5), trend='n')

# varmax : mmade modelinte peru,, other example , ARIMA
# order=(5,5) : ee 5,5 inte idea vananth munnathe graphil nnu aanu, 
# athayath,plot 1 and 2 lu ...5 th point thott aanu blue region lu poyath
# blue region nthannu ariyila
# the reason to take sales and customers : ith randum time ansarich maravunnathalle..so athinte time series analysis namma edkunnu
# sales predict cheyan
# exog : athikam variantion time ine depend cheyathath..angane onnula..chumma


res = mod.fit(maxiter=1000, disp=False)
print(res.summary())
#Note the AIC value - lower AIC => better model


# In[57]:


ax = res.impulse_responses(30, orthogonalized=True).plot(figsize=(13,3))
ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');


# In[58]:


# just like train test split
train_len = 900
train_data = train_store_1[0:train_len] 
test_data = train_store_1[train_len:]

print("Train data: {}".format(train_data.shape))
print("Test data: {}".format(test_data.shape))


# In[59]:


start_index = test_data.index.min()
end_index = test_data.index.max()
predictions = res.predict(start=start_index, end=end_index, exog=[''])


# In[60]:


plt.figure(figsize=(96, 16)) 
plt.plot( train_data['Sales'], label='Train')
plt.plot(test_data['Sales'], label='Test')
plt.plot(predictions['Sales'], label='VARMAX')
plt.legend(loc='best')
plt.title('VAR Model - Sales')
plt.show()


# In[61]:


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


# In[ ]:


store : 1
mode : varmax
variable : sales
Rmse : 
mape : 


# In[ ]:




