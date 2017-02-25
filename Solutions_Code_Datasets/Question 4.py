import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2

# TODO: Read data set in a pandas Dataframe and print th first 5 values and generate descriptive statistics for the numberical features
dataset = pd.read_excel('AKQA_Dataset_Test.xlsx')
dataset = dataset.drop('LastSaleDate', axis=1)
dataset['age'] = 2014 - dataset['YearBuilt']
dataset_dimensions = dataset.shape
print 'The dataset consists of: \n {} listings \n {} features for each listing \n'.format(dataset_dimensions[0], dataset_dimensions[1])
print dataset.describe()

# Pick the Features that will be used for the classification
features = {'ListPrice': dataset['ListPrice'], 'BEDS': dataset['BEDS'], 'LotSize': dataset['LotSize'], 'SQFT': dataset['SQFT']}
dataset_corr = pd.DataFrame(features)

print dataset_corr.describe()

# Remove listings where Lotsize and BATHS are not = NULL
if dataset_corr.isnull().values.any():
    dataset_corr = dataset_corr.dropna()

print dataset_corr.describe()

# Correlate every feature with the list price and decide
ListPrice = dataset_corr['ListPrice']
print np.corrcoef(dataset_corr['BEDS'], ListPrice)
print np.corrcoef(dataset_corr['LotSize'], ListPrice)
print np.corrcoef(dataset_corr['SQFT'], ListPrice)








