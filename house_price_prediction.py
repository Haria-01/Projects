#!/usr/bin/env python
# coding: utf-8

# ### Importing the major libraries

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams['figure.figsize']=(20,10)
import seaborn as sns


# ### Importing the data

# In[2]:


df1=pd.read_excel("DS - Assignment Part 1 data set.xlsx" )
df1=df1.drop(columns=['Transaction date'],axis=1)
print(df1.shape)
df1.head()


# Transaction date column has been removed since it does not provide any useful information regarding the price of the house.

# In[3]:


df1.columns


# ### Checking for null values

# In[4]:


df1.isnull().sum()


# In[5]:


df1.describe()


# In[6]:


corr=df1.corr(method='spearman')
plt.figure(figsize=(15,15))
g=sns.heatmap(corr,annot=True)


# ### from the above correlation heatmap it can be concluded that the
# ### 1)House price is having positive correlation with number of convenience stores, longitue and latitude. 
# ### 2)House price is having negative correlation with distance from nearest metro station and House age
# ### 3)House price is having neutral correlation with number of bedrooms,house size.

# ### Importing the major regression algorithms for tarining the models and to make predictions

# In[7]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# ### Features to evaluate the House price prediction are stored in the variable X and the Target Price of House is stored in the variable y

# In[8]:


data=df1.values
X,y=data[:,:-1],data[:,-1]


# ## Procedure for selecting the final model for House price prediction:
# ### Without removing the outliers:
# As the dataset is very small  we will evaluate the various regression models using Repeated kfold cross validation method.
# Here we will use mean absolute error metric for the comparion between various regression models.
# 
# ### Perform the above step again by removing the outliers:
# Now compare the models of with outliers and the models without outliers and select the best model.
# Based on the best model we will perform hyper parameter tuning and then finalize the model for House price prediction.
# 
# 

# ### LinearRegression Model

# In[9]:


#define model
model=LinearRegression()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### ElasticNet Regressor Model

# In[10]:


#define model
model=ElasticNet(alpha=1,l1_ratio=0.5)
DecisionTree_model=DecisionTreeRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### DecisionTreeRegressor Model

# In[11]:


#define model
model=DecisionTreeRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### RandomForestRegressor Model

# In[12]:


#define model
model=RandomForestRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### AdaBoostRegressor Model

# In[13]:


#define model
model=AdaBoostRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### GradientBoostingRegressor Model

# In[14]:


#define model
model=GradientBoostingRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### Support vector regressor Model

# In[15]:


#define model
model=SVR()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### CatBoostRegressor Model

# In[16]:


#define model
model=CatBoostRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### XGBRegressor Model

# In[17]:


#define model
model=XGBRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### Removing the outliers from the dataset i.e. including the target variable because this is a typical case of a sampling problem in which we have to remove the outliers from the target variable aswell.
# ### Standardizing  both the input and output variables which is important for the models like Support vetor machines, linear models like linear, lasso, ridge and ElasticNet

# In[18]:


df=df1.copy()
numerical_features=[feature for feature in df.columns if df[feature].dtype!='O']
discreate_features=[feature for feature in df.columns if len(df[feature].unique())<25]
contineous_features=[feature for feature in numerical_features if feature not in discreate_features]
categorical_features=[feature for feature in df.columns if feature not in numerical_features]
print('Numerical_features count: {}'.format(len(numerical_features)))
print('Discreate_features count: {}'.format(len(discreate_features)))
print('Contineous_features count: {}'.format(len(contineous_features)))
print('Categorical_features count: {}'.format(len(categorical_features)))


# ### Box plot to detect the outlier in the dataset

# In[19]:


for feature in df.columns:
    sns.boxplot(df[feature])
    plt.xlabel(feature)
    plt.title(feature)
    plt.figure(figsize=(10,10))
    plt.show


# ### Using interquartile range method we will remove the outliers

# In[20]:


for feature in df.columns:
    IQR=df[feature].quantile(0.75)-df[feature].quantile(0.25)
    ub=df[feature].quantile(0.75)+(1.5*IQR)
    lb=df[feature].quantile(0.25)-(1.5*IQR)
    df.loc[df[feature]>ub,feature]=ub
    df.loc[df[feature]<lb,feature]=lb


# ### Box plots after the removal of the outliers

# In[21]:


for feature in df.columns:
    sns.boxplot(df[feature])
    plt.xlabel(feature)
    plt.title(feature)
    plt.figure(figsize=(10,10))
    plt.show


# ### Input features are stored in the X variable
# ### Target House Price is stored in the y variable

# In[22]:


data=df.values
X,y=data[:,:-1],data[:,-1]


# ###  Data preparation for model evaluation with k-fold cross-validation
# 1)Define the Pipeline
# 
# 2)Define the evaluation procedure
# 
# 3)Evaluate the model using cross-validation
# 
# 4)Report performance
# 
# 
# 

# In[23]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ### LinearRegression Model by removing outliers and  standardization of the dataset

# In[24]:


#define the pipeline
steps=list()
steps.append(('scaler',StandardScaler()))
steps.append(('model',LinearRegression()))
pipeline=Pipeline(steps=steps)
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(pipeline,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### ElasticNet Regressor Model  by removing outliers and  standardization of the dataset

# In[25]:


#define the pipeline
steps=list()
steps.append(('scaler',StandardScaler()))
steps.append(('model',ElasticNet(alpha=1,l1_ratio=0.5)))
pipeline=Pipeline(steps=steps)
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(pipeline,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### Support vector regressor Model by removing outliers and  standardization of the dataset

# In[26]:


#define the pipeline
steps=list()
steps.append(('scaler',StandardScaler()))
steps.append(('model',SVR()))
pipeline=Pipeline(steps=steps)
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(pipeline,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### DecisionTreeRegressor model  by removing outliers and without standardization of the dataset

# In[27]:


#define model
model=DecisionTreeRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### RandomForestRegressor model  by removing outliers and without standardization of the dataset

# In[28]:


#define model
model=RandomForestRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### AdaboostRegressor model  by removing outliers and without standardization of the dataset

# In[31]:


#define model
model=AdaBoostRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### GradientboostRegressor model  by removing outliers and without standardization of the dataset

# In[32]:


#define model
model=GradientBoostingRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### CatBoostRegressor model  by removing outliers and without standardization of the dataset

# In[33]:


#define model
model=CatBoostRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### XGBoostRegressor model  by removing outliers and without standardization of the dataset

# In[36]:


#define model
model=XGBRegressor()
#define model evaluation method
cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=1)
#Evaluate of model
scores=cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1)
#force scores to be positive
scores=np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# ### Out of all the models GradientBoostRegressor gave the best Mean Absolute Error score i.e 4.823 and the second best is RandomForestRegressor
# 
# ### So now we will perform hyperparameter tuning on GradientBoostRegressor model 

# ### Optimal parameters(Hyperparameter tuning) for GradientBoostRegressor using GridSearchCV for Regression

# In[37]:


data=df.values
X,y=data[:,:-1],data[:,-1]


# In[39]:


# grid searching key hyperparameters for gradient boosting 

from sklearn.model_selection import GridSearchCV
# define the model with default hyperparameters
model = GradientBoostingRegressor()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
grid['subsample'] = [0.5, 0.7, 1.0]
grid['max_depth'] = [3, 7, 9]
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='neg_mean_absolute_error')
# execute the grid search
grid_result = grid_search.fit(X, y)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# # Final GradientBoostRegressor model for House Price Prediction

# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data=df.values
X,y=data[:,:-1],data[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
model=GradientBoostingRegressor(learning_rate=0.01,max_depth=3,n_estimators=500,subsample=0.7)
model.fit(X_train,y_train)
yhat=model.predict(X_test)
MAE=mean_absolute_error(y_test,yhat)
MAE


# In[ ]:





# In[ ]:




