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
from sklearn.ensemble import RandomForestClassifier

# TODO: Read data set in a pandas Dataframe and print th first 5 values and generate descriptive statistics for the numberical features
dataset = pd.read_excel('AKQA_Dataset_edited.xlsx') # Read the dataset with ordered realty names and grouped realty subdivisions
dataset = dataset.drop('LastSaleDate', axis=1)
dataset['age'] = 2014 - dataset['YearBuilt']
dataset_dimensions = dataset.shape
print 'The dataset consists of: \n {} listings \n {} features for each listing \n'.format(dataset_dimensions[0], dataset_dimensions[1])

# Question 1

# Group the needed features togeather
data = {'Realty': dataset['Realty'], 'SoldPrev': dataset['SoldPrev'], 'ShortSale': dataset['ShortSale']}
q1data = pd.DataFrame(data)
print q1data.head(5)
print '\n', q1data.describe()

# Check if any of the features have empty values
print '\n Null Value Check = {}'.format(q1data.isnull().values.any())

# Group the subdivisions of the companies togeather
Realties = q1data['Realty']
Realties_unique = Realties.unique()
#print '\n Here are the Realties available in the Cities:', '\n', Realties_unique, '\n'
print 'There are {} unique Realties'.format(len(Realties_unique))

print Realties_unique

# Create a Dataframe for all companies vs number of listings, SoldPrev, DOM, ShortSale
realty_statistcs = pd.DataFrame(index=Realties_unique, columns=['nListings', 'SoldPrev', 'ShortSale'])

for realty in Realties_unique:
    for feature in realty_statistcs.columns:
        if feature == 'nListings':
            value_c = q1data[(q1data.Realty == realty)].count().get_value('Realty')
            realty_statistcs.set_value(realty, feature, value_c)
        elif feature == 'SoldPrev':
            value_c = q1data[(q1data.Realty == realty) & (q1data.SoldPrev == 'Y')].count().get_value('Realty')
            realty_statistcs.set_value(realty, feature, value_c)
        elif feature == 'ShortSale':
            value_c = q1data[(q1data.Realty == realty) & (q1data.ShortSale == 'Y')].count().get_value('Realty')
            realty_statistcs.set_value(realty, feature, value_c)
print realty_statistcs.head(5)

# Plot the statistics for all the realties
plt.style.use('ggplot')
realty_statistcs.plot.bar(stacked=True)
plt.title("Statistics for Realties", fontsize='10')
plt.savefig("Realties statistics.jpg", dpi=300)
plt.close()

# Filter Companies according to the number of listings > 90
realty_statistcs = realty_statistcs[realty_statistcs.nListings > 90]

print realty_statistcs
realty_statistcs.plot.bar(stacked=True, figsize=(15, 10), fontsize=12)
plt.title("Statistics for Top Realties", fontsize='10')
plt.ylabel('Counts')
plt.savefig("Top Realties statistics.jpg", dpi=300)
plt.close()


















