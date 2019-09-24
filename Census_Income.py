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
raw_data['education'].replace(('10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th',
                            '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool',
                            'Prof-school', 'Some-college'), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                            inplace =True)

print('End!!!')