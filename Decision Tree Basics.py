# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:00:02 2021

@author: Aben George
"""


# SPLIT 1 - Analyse data set and split based on possible entropy levels to 
#           improve end result performance

import pandas as pd

# PREPROCESSING

df = pd.read_csv('salaries.csv')

# SPLIT dataframe between independant and dependant variables

inputs = df.drop("salary_more_then_100k", axis= 'columns')
target = df['salary_more_then_100k']

# CONVERT categorical variables using LabelEncoder from Sklearn

from sklearn.preprocessing import LabelEncoder

# Create individual objects for cat variables

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# these encoded values need to Fit and Transform and placed back into input df

# use .fit_transform 

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Create new df with encoded only

inputs_n = inputs.drop(['company', 'job', 'degree'], axis = 'columns')

# need to specify axis on columns

#-----------------------------------------------------------------------

# MODELLING

from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target) # not test-train splitting, but ideally you should

# Test - but need to use categorical variable encoded values

print(model.predict([[2,2,1]]))



