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

print 'The descriptive statistics for the numerical values in the dataset are as follows: \n {}'.format(dataset.describe())
# Get data between the 25th percentile and 75th percentile
ListPrice_25 = dataset.ListPrice.quantile(0.25)
ListPrice_75 = dataset.ListPrice.quantile(0.75)


ListPrice_bet2575_andbyzip = dataset[((dataset.ListPrice > ListPrice_25) & (dataset.ListPrice < ListPrice_75)) & ((dataset.ZIP == 55104) | (dataset.ZIP == 55108))]
ListPrice_zip = dataset[(dataset.ZIP == 55104) | (dataset.ZIP == 55108)]

print ListPrice_bet2575_andbyzip.shape
print ListPrice_zip.shape

