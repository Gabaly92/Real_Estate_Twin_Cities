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

# TODO: Read data set in a pandas Dataframe and print th first 5 values and generate descriptive statistics for the numberical features
dataset = pd.read_excel('AKQA_Dataset_Test.xlsx')
dataset = dataset.drop('LastSaleDate', axis=1)
dataset['age'] = 2014 - dataset['YearBuilt']
dataset_dimensions = dataset.shape
print 'The dataset consists of: \n {} listings \n {} features for each listing \n'.format(dataset_dimensions[0], dataset_dimensions[1])
#print 'The descriptive statistics for the numerical values in the dataset are as follows: \n {}'.format(dataset.describe())



# Question 2

# TODO: Generate scatter plot for the houses square footage vs the listing prices

# Get sqft and listing_prices data from dataset
sqft = np.array(dataset['SQFT']).reshape(-1,1)
listing_prices = np.array(dataset['ListPrice']).squeeze()

print np.shape(sqft)
print np.shape(listing_prices)

# Function for determining the performance of a classifier
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    score = r2(y_true, y_predict)

    # Return the score
    return score

# Fuction for fitting model on data with cross validation and parameter tuning
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)

    # TODO: Create a polynomial regression object
    regressor = make_pipeline(PolynomialFeatures(), Ridge())
    print regressor.get_params().keys()

    # TODO: Create a dictionary for the parameter 'degrees' with a range from 1 to 10
    params = {'polynomialfeatures__degree': [1, 2, 3 , 4 , 5 , 6, 7, 8, 9, 10]}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Find best estimator with best tuned parameters
best_estimator = fit_model(sqft, listing_prices)
print '\n', best_estimator

# Get Prediction with the best estimator
predicted_listing_prices = best_estimator.predict(sqft)

# Plot scatter plot of the data
plt.scatter(sqft, listing_prices, s=10, alpha=0.3)

# plot predicted listing prices on the same plot
plt.plot(sqft, predicted_listing_prices)
plt.title("Twin Cities Real Estate Price", fontsize='10')
plt.xlabel('SQFT (ft)', fontsize='10')
plt.ylabel('Listing Price ($)', fontsize='10')
plt.show()
plt.savefig("TwinCities_real_estate_polynomial_regression.jpg", dpi=300)
plt.close()

print 'The listing price for a 2111 square foot house is ${}'.format(best_estimator.predict(2111))



















