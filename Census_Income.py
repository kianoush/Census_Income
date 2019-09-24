"""
Census Income
Predict if an individual makes greater or less than $50000 per year
"""

"""
Instances: 48842

Attributes: 15

Tasks: Classification

Downloads: 1275

Year Published: 1996

Missing Values: Yes
"""

import pandas as pd
import torch
import torch.nn as nn
import missingno
import matplotlib.pyplot as plt
import numpy as np



"""
Import data
"""

raw_data = pd.read_csv('census_income_dataset.csv')
raw_data.info()
print(raw_data.describe())
columns = raw_data.keys()
# missingno.matrix(raw_data, figsize = (10,5))
# plt.show()

for column in columns:
    if raw_data[column].dtype == object:
        print(column, ':', len(np.unique(raw_data[column])))
        print(np.unique(raw_data[column]), end='')
    print()

for column in columns:
    raw_data[column].replace(('?'), (np.nan), inplace=True)
    print(column, " :", raw_data[column].isna().sum())

raw_data['workclass'].fillna('no_fill', inplace=True)
raw_data['occupation'].fillna('no_fill', inplace=True)


raw_data['workclass'].replace(('no_fill', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
                            'Self-emp-not-inc', 'State-gov', 'Without-pay'), (0, 1, 2, 3, 4, 5, 6, 7, 8),
                            inplace =True)
print(raw_data['education'].value_counts().sort_index())
raw_data['education'].replace(('10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th',
                            '9th'), ('school'),
                            inplace =True)
raw_data['education'].replace(('school', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool',
                            'Prof-school', 'Some-college'), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                            inplace =True)
raw_data['marital_status'].replace(('Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'), (0, 1, 2, 3, 4, 5, 6),
                            inplace =True)
raw_data['occupation'].replace(('?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                            'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                            'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
                            'Tech-support', 'Transport-moving'), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                            inplace =True)
raw_data['occupation'].replace(('?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                            'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                            'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
                            'Tech-support', 'Transport-moving'), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                            inplace =True)

raw_data['sex'] = np.where(raw_data.sex=='Male', 0, 1)
raw_data['income_level'] = np.where(raw_data.income_level=='<=50K', 0, 1)

d = (raw_data.hours_per_week.value_counts())


# raw_data.drop(column=['fnlwgt', 'education_num',
#        'race', 'capital_gain', 'capital_loss', 'native_country'])

print('End!!!')