import sklearn
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

np.random.seed(42)  # set seed for deterministic data generation
data = np.random.multivariate_normal(mean=[5, 5], cov=[[3, 8], [4, 8]], size=500)
outlier = np.random.multivariate_normal(mean=[7, 17], cov=[[2, 1], [1, 2]], size=50)
X = np.concatenate([data[:, 0], outlier[:, 0]])  # first dimension are data points
y = np.concatenate([data[:, 1], outlier[:, 1]])  # second dimension are values


def visualize_data(X: np.ndarray, y: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, alpha=0.7, edgecolors='g')
    plt.show()


visualize_data(X, y)


# estimate regression line beta_hat
def estimate_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(X.shape[0]), X]  # Concatenate a column of ones to X
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T = transpose of X, @ = matrix multiplication
    return beta_hat


beta_hat = estimate_beta(X, y)


# use linear regression to compute predictions
def compute_predictions(X: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones to X
    return X @ beta_hat


predictions = compute_predictions(X, beta_hat)


# calculate mean squared error
def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return mean_squared_error(y_true, y_pred)


def visualize_predictions(X: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, alpha=0.7, edgecolors='g')
    plt.plot(X, predictions, color='r')
    plt.show()


visualize_predictions(X, y, predictions)
print("Mean Squared Error: ", compute_mse(y, predictions))