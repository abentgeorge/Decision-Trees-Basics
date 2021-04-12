# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:15:00 2021

@author: Aben George
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

#Warning Issue Solution
pd.options.mode.chained_assignment = None  # default='warn'


df = pd.read_csv('titanic.csv')

target = df['Survived']
inputs = df[['Pclass', 'Sex', 'Age', 'Fare']] #Selecting multiple columns needs an extra []

le_pclass = LabelEncoder()
inputs['pclass_n'] = le_pclass.fit_transform(inputs['Pclass'])


le_sex = LabelEncoder()
inputs['sex_n'] = le_sex.fit_transform(inputs['Sex'])

inputs_n = inputs.drop(['Pclass', 'Sex'], axis = 'columns')

# DEALING WITH NA IN AGE< TAKING MEAN

import math

meanage= math.floor(inputs_n['Age'].mean())

inputs_n['Age'] = inputs_n['Age'].fillna(29)
#---------------------------------------------------------------

model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target)

print(model.predict([[55,16,1,0]]))
print(model.score(inputs_n, target))