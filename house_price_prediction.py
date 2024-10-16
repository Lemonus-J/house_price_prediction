#!/usr/bin/env python
# coding: utf-8

# In[277]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jlb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
plt.style.use('ggplot')


# In[181]:


df = pd.read_excel('../dataset/HousePricePrediction.xlsx')


# # Data Understanding
# 
# Look at the basic information of our data set to get a rough idea of what we are dealing with

# In[182]:


df.columns


# In[183]:


df.head(5)


# In[184]:


df.shape


# # Data Cleaning
# 
# Do basic data cleaning to remove obvious errors from our data set

# In[ ]:


# remove the Id column as it wont be useful in finding any data
df.drop(['Id'], axis=1, inplace=True)


# In[217]:


# deciding what to do with null target variables depends on the amount of null values:
# - if the number of null values is large, consider dropping them as replacing them with
#   a mean value might create a biased dataset
# - if the number of null values is small, we can replace them with mean values to
#   to make the data distribution symmetric

# in this case, only 12 our of 2919 rows are null, so we can replace them with mean value
# len(list(df[df['SalePrice'].isnull()]))
df['SalePrice'] = df['SalePrice'].fillna(df['SalePrice'].mean()) 


# In[230]:


df = df.dropna()


# In[224]:


is_object = df.dtypes == 'object'
object_cols = list(is_object[is_object].index)
print('Categorial variables : ', len(object_cols))

is_int = df.dtypes == 'int'
int_cols = list(is_int[is_int].index)
print('Integer variables : ', len(int_cols))

is_float = df.dtypes == 'float'
float_cols = list(is_float[is_float].index)
print('Float variables : ', len(float_cols))


# In[225]:


# find correlation between all numerical values
df_corr = df.select_dtypes(include =['number'])
sns.heatmap(
    df_corr.dropna().corr(), 
    annot=True, 
    fmt='.2f',)
# plt.show()


# In[226]:


# After observing the hearmap above, I am interested to see the relationship 
# between 3 values that has the highest correlation.
plt.figure(figsize=(25, 10))
df_hasBsmt = df.query('TotalBsmtSF > 0')
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', hue='YearRemodAdd', data=df_hasBsmt)
# plt.show()


# # Rough Observation
# 
# Basement size and Remodel/Construction date might determine Sale Price

# In[227]:


# Identify how many unique categorial values we have in each column
unique_values = []
for col in object_cols:
    unique_values.append(df[col].unique().size)
plt.title('No. Unique values of Categorical Features')
sns.barplot(x=object_cols, y=unique_values)
# plt.show()


# In[228]:


# identify and plot actual count of each categorial column
plt.figure(figsize=(24, 5)) # draws a 24 x 5 background
plt.xticks([]) # hides the x labels
plt.yticks([]) # hides the y labels

index = 1

for col in object_cols:
    y_data = df[col].value_counts()
    plt.subplot(1, 4, index)
    plt.xticks(rotation=90)# rotates the xlabels by 90 degrees
    ax = sns.barplot(x=list(y_data.index), y=y_data)
    ax.set_title(col)
    index += 1

# plt.show()


# We will need to convert the categorial values into binary vectors in order to load them into the ML model. To do this, we will use OneHotEncoder to transform the values

# In[ ]:


# create OneHotEncoder object
'''
handle_unknown: Decides what happens when the encoder encounters a category that wasn’t seen during training.
    - 'error' (default): Raises an error if an unknown category is found.
    - 'ignore': Skips the unknown category and outputs a row of zeros for it.

sparse_output: Controls whether the encoded output is returned as a sparse matrix (which saves memory for large datasets with lots of zeros).
    - True (default): Returns a sparse matrix.
    - False: Returns a regular dense array.
'''
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)


# In[ ]:


# to understand what fit_transform() does, we first need to see what fit() and transform() does by themselves
'''
1. fit()
When you call fit(), the encoder learns the unique categories present in the dataset for each feature.
This essentially "trains" the encoder to understand the categories that need to be one-hot encoded.

2. transform()
Once the encoder is fitted, calling transform() applies the encoding to the dataset.
For each category in the dataset, it creates a binary vector where:
    - The position corresponding to that category is marked as 1.
    - All other positions are 0.
    
3. fit_transform()
The fit_transform() method combines these two steps. 
It fits the encoder to the data and then immediately transforms the data based on the learned categories. 
This is a common shortcut used when you don’t need to separate fitting and transforming into two steps.
'''
oh_cols = pd.DataFrame(ohe.fit_transform(df[object_cols]))


# In[ ]:


# after populating the data frame, we then fill in the index and columns
oh_cols.index = df.index
oh_cols.columns = ohe.get_feature_names_out()


# In[271]:


# after that, we drop the existing categorial columns and concat the new training columns to form a new DataFrame
df_final = df.drop(object_cols, axis=1)
df_final = pd.concat([df_final, oh_cols], axis=1)


# # Preparing our final DataFrame to train the model
# 
# after we've finally processed our data, we can finally load them into a training model

# In[272]:


X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice'] # target variable

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=.8, test_size=.2, random_state=0)
# X_valid


# In[273]:


# X_train


# In[278]:


model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))


jlb.dump(model_LR, "./model/lr_model.sav")
