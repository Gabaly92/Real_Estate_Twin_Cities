import pandas as pd

# TODO: Read data set in a pandas Dataframe and print th first 10 values and
dataset = pd.read_excel('AKQA_Dataset_Test.xlsx')
dataset_dimensions = dataset.shape
print 'The dataset consists of: \n {} listings \n {} features for each listing \n'.format(dataset_dimensions[0], dataset_dimensions[1])
print 'The descriptive statistics for the numerical values in the dataset are as follows: \n {}'.format(dataset.describe())

