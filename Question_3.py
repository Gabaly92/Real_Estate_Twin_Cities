import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score as r2
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# TODO: Read data set in a pandas Dataframe and print th first 5 values and generate descriptive statistics for the numberical features
dataset = pd.read_excel('AKQA_Dataset_Test.xlsx')
dataset = dataset.drop('LastSaleDate', axis=1)
dataset['age'] = 2014 - dataset['YearBuilt']
dataset_dimensions = dataset.shape
print 'The dataset consists of: \n {} listings \n {} features for each listing \n'.format(dataset_dimensions[0], dataset_dimensions[1])
print dataset.describe()

# Pick the Features that will be used for the classification
features = {'ListPrice': dataset['ListPrice'], 'BEDS': dataset['BEDS'], 'BATHS': dataset['BATHS'], 'SQFT': dataset['SQFT'], 'age': dataset['age'], 'SoldPrev': dataset['SoldPrev'],
            'ZIP': dataset['ZIP']}
dataset_new = pd.DataFrame(features)

# Get training set from datase_new, it will include all zips
ListPrice_25 = dataset_new.ListPrice.quantile(0.25)
ListPrice_75 = dataset_new.ListPrice.quantile(0.75)

train = dataset_new[((dataset_new.ListPrice > ListPrice_25) & (dataset_new.ListPrice < ListPrice_75)) & ((dataset_new.ZIP != 55104) & (dataset_new.ZIP != 55108))]

train_x = train.drop(['ZIP','SoldPrev',], axis=1)
train_y = train['SoldPrev']
print type(train_y)
train_y = train_y.replace(['Y', 'N'], [1, 0])

# Get data between the 25th percentile and 75th percentile and in ZIPS 55104 55108


zip104 = dataset_new[((dataset_new.ListPrice > ListPrice_25) & (dataset_new.ListPrice < ListPrice_75)) & (dataset_new.ZIP == 55104)]
zip104 = zip104.drop(['ZIP', 'SoldPrev'], axis=1)
print 'Zip 55104 test set \n'
print zip104.columns.values
print zip104.describe()

zip108 = dataset_new[((dataset_new.ListPrice > ListPrice_25) & (dataset_new.ListPrice < ListPrice_75)) & (dataset_new.ZIP == 55108)]
zip108 = zip108.drop(['ZIP', 'SoldPrev'], axis=1)
print 'Zip 55108 test set \n'
print zip108.columns.values
print zip108.describe(), '\n'

# get 10 random values from zip 55104 to make a fair test
zip104 = zip104.sample(10)

print 'Zip 55104 10 samples description', zip104.describe()

# Train Random Forest on the training set
rfc = RandomForestClassifier(n_estimators= 50)
rfc.fit(train_x, train_y)

# Predict Selling for the two zip codes
zip104_pred = rfc.predict(zip104)
print 'predictions for zip 55104 \n {}'.format(zip104_pred)

zip108_pred = rfc.predict(zip108)
print 'predictions for zip 55108 \n {}'.format(zip108_pred)








