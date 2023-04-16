"""
EDA
train = pd.read_csv("...\\train.csv")

train.head(10): we see that the empty cells are in NaN form

len(train["price_CHF"].value_counts("NaN") outputs 631 out of 900 observation.
A substantial proportion of our target variable contains missing numbers.

train.isna().sum(axis=1).value_counts()
3 900
Every row has exactly three missing values

train.isna().sum()
season         0
price_AUS    262
price_CHF    269
price_CZE    268
price_GER    269
price_ESP    269
price_FRA    264
price_UK     287
price_ITA    266
price_POL    265
price_SVK    281


"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer #NoQA
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import r2_score
from sklearn import preprocessing

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')

    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))
    print('\n')

    # Dummy (numpy) initialization of the X_train, X_test and y_train
    X_train = np.zeros_like(train_df.drop(['price_CHF'], axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # Convert categorical variable "season" into 4 indicator (0/1) variables
    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

    # Standardization of dependent variables
    train_df_y = train_df["price_CHF"]
    train_df_x = train_df.drop(columns="price_CHF")

    train_df_x = preprocessing.scale(train_df_x)
    test_df = preprocessing.scale(test_df)

    # Initialize class
    #imp = IterativeImputer(missing_values=np.nan)
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = KNNImputer(n_neighbors=35)
                                                                                                   
    # Convert to numpy
    #train_df = train_df.dropna(subset="price_CHF")
    #train_df = train_df.to_numpy()
    #test_df = test_df.to_numpy()

    # Data imputation
    train_df_y = train_df_y.to_numpy()
    train_df = np.column_stack((train_df_x, train_df_y))
    train_df = imp.fit_transform(train_df)
    y_train = train_df[:,-1]
    X_train = np.delete(train_df, -1, axis=1)

    X_test = imp.fit_transform(test_df)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (
                X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred = np.zeros(X_test.shape[0])
    # TODO: Define the model and fit it using training data. Then, use test data to make predictions

    kernel = kernels.Sum(kernels.DotProduct(), kernels.RBF())
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gpr.fit(X_train, y_train)
    print(gpr.score(X_train, y_train))
    y_pred, sigma = gpr.predict(X_test, return_std=True)


    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

