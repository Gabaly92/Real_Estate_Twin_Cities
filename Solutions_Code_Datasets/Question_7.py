import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score as r2
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

#TODO: Read data set in a pandas Dataframe and print th first 5 values and generate descriptive statistics for the numberical features
dataset = pd.read_excel('AKQA_Dataset_edited.xlsx')
dataset = dataset.drop('LastSaleDate', axis=1)
dataset['age'] = 2014 - dataset['YearBuilt']
dataset_dimensions = dataset.shape
print 'The dataset consists of: \n {} listings \n {} features for each listing \n'.format(dataset_dimensions[0], dataset_dimensions[1])
print 'Dataset before removing Null listings: \n', dataset.describe()

# remove NULL listing from the data set
dataset = dataset.dropna()
print 'Dataset after removing Null listings: \n', dataset.describe()
print dataset.describe()

# convert Categorial features into dummies and Yes No features into binary values
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output
dataset_pp = preprocess_features(dataset)

# Create Training Set and Test Set
Target_Prices = dataset_pp['ListPrice']
dataset_pp = dataset_pp.drop(['ListPrice', 'ID'], axis=1)

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(dataset_pp, Target_Prices, test_size=0.3,random_state=10)

# Create and fit classifier to data
rfr = RandomForestRegressor(200)
rfr.fit(X_train, y_train)

# print the feature importances
feature_importance = list(rfr.feature_importances_)
feature_importance_ind = sorted(range(len(feature_importance)), key=lambda k: feature_importance[k],reverse=True)
features = list(dataset_pp.columns.values)

# Create a dataframe with for the feature importances
features_values_print = []
features_values_names = []
n = 10
for i in range(0,n):
    features_values_print.append(feature_importance[feature_importance_ind[i]])
    features_values_names.append(features[feature_importance_ind[i]])

print features_values_print
print features_values_names








