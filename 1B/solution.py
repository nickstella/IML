# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))

    X_transformed[:, :5] = X

    X_2 = np.array(X)

    for i in range(X_2.shape[0]):
        for j in range(X_2.shape[1]):
            X_2[i][j] = X_2[i][j]**2

    X_transformed[:,5:10] = X_2

    X_3 = np.array(X)

    for i in range(X_3.shape[0]):
        for j in range(X_3.shape[1]):
            X_3[i][j] = np.exp(X_3[i][j])

    X_transformed[:,10:15] = X_3

    X_4 = np.array(X)

    for i in range(X_4.shape[0]):
        for j in range(X_4.shape[1]):
            X_4[i][j] = np.cos(X_4[i][j])

    X_transformed[:,15:20] = X_4

    X_5 = np.ones(X.shape[0])

    X_transformed[:,20] = X_5

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)

    #np.savetxt("./resultssss.csv", X_transformed, fmt="%.12f")

    # Basic linear regression with sklearn
    #linear_regression = LinearRegression(fit_intercept=False)
    #linear_regression.fit(X_transformed, y)


    # Explicit solve, with np matrix inversion
    #w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_transformed.transpose(), X_transformed)), X_transformed.transpose()), y)

    w = np.linalg.solve(np.matmul(X_transformed.transpose(), X_transformed), np.matmul(X_transformed.transpose(), y))

    assert w.shape == (21,)
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
