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
    raw_data[column].replace(('?'), (np.nan), inplace = True)
    print(column," :",raw_data[column].isna().sum())




print('End!!!')