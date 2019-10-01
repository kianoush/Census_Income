"""
Census Income
Predict if an individual makes greater or less than $50000 per year
"""

"""
Instances: 48842

Attributes: 15

Tasks: Classification

Year Published: 1996

Missing Values: Yes
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split


"""
Import data
"""

raw_data = pd.read_csv('census_income_dataset.csv')
raw_data.info()
#print(raw_data.describe())
columns = raw_data.keys()
# missingno.matrix(raw_data, figsize = (10,5))
# plt.show()

for column in columns:
    if raw_data[column].dtype == object:
        print(column, ':', len(np.unique(raw_data[column])))
        #print(np.unique(raw_data[column]), end='')
    #print()

for column in columns:
    raw_data[column].replace(('?'), (np.nan), inplace=True)
    #print(column, " :", raw_data[column].isna().sum())

raw_data['workclass'].fillna('no_fill', inplace=True)
raw_data['occupation'].fillna('no_fill', inplace=True)


raw_data['workclass'].replace(('no_fill', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
                            'Self-emp-not-inc', 'State-gov', 'Without-pay'), (0, 1, 2, 3, 4, 5, 6, 7, 8),
                            inplace =True)
#print(raw_data['education'].value_counts().sort_index())
raw_data['education'].replace(('10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th',
                            '9th'), ('school'),
                            inplace =True)
raw_data['education'].replace(('school', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool',
                            'Prof-school', 'Some-college'), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                            inplace =True)
raw_data['marital_status'].replace(('Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
                            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'), (0, 1, 2, 3, 4, 5, 6),
                            inplace =True)
raw_data['occupation'].replace(('no_fill', 'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                            'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                            'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
                            'Tech-support', 'Transport-moving'), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                            inplace =True)

raw_data['sex'] = np.where(raw_data.sex == 'Male', 0, 1)
raw_data['income_level'] = np.where(raw_data.income_level == '<=50K', 0, 1)
raw_data['hours_per_week'] = np.where(raw_data.hours_per_week < 40, 0, 1)

raw_data.drop(columns=['fnlwgt', 'education_num',
       'race', 'capital_gain', 'capital_loss', 'native_country', 'relationship'], inplace=True)

# raw_data.info()
# print(np.unique(raw_data['occupation']))


data = raw_data.iloc[:, :7].values
labels = raw_data.iloc[:, 7].values



"""
Split train and test
"""
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
X_valid = torch.tensor(X_valid).float()

Y_train = torch.tensor(Y_train).long()
Y_test = torch.tensor(Y_test).long()
Y_valid = torch.tensor(Y_valid).long()

"""
Model
"""
num_features = X_train.shape[1]
num_classes = 2
num_hiddenl = 10
model = torch.nn.Sequential(torch.nn.Linear(num_features, num_hiddenl),
                            torch.nn.ReLU(),
                            torch.nn.Linear(num_hiddenl, num_classes))

"""
Optim
"""
optim = torch.optim.Adam(model.parameters(), lr=0.001)

"""
Loss
"""
loss = torch.nn.CrossEntropyLoss()

"""
train
"""

num_sample_train = torch.tensor(X_train.shape[0])
num_sample_test = torch.tensor(X_test.shape[0])
num_sample_valid = torch.tensor(X_valid.shape[0])

num_epochs = 200
for epoch in range(num_epochs):
    optim.zero_grad()
    Y_pred = model(X_train)
    loss_value = loss(Y_pred, Y_train)

    num_corrects = torch.sum(torch.max(Y_pred, 1)[1] == Y_train)
    acc_train = num_corrects.float() / float(num_sample_train)

    loss_value.backward()
    optim.step()

    yp = model(X_valid)
    num_corrects = torch.sum(torch.max(yp, 1)[1]==Y_valid)
    acc_valid = num_corrects.float() / float(num_sample_valid)
    print("Epoch: ", epoch, 'Train Loss: ', loss_value.item(), 'Train Accurecy: ', acc_train.item(), 'VALIDATION acoreccy', acc_valid.item())



yp = model(X_test)
num_corrects = torch.sum(torch.max(yp, 1)[1]==Y_test)
acc_test = num_corrects.float() / float(num_sample_test)
print('Test Accurecy: ', acc_test.item())

print('End!!!')