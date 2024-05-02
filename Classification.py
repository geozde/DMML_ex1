import sklearn
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

# x = 2D array with coordinates of data points
# y = label that specifies to which of the blobs a data point belongs
X, y = make_blobs(500, centers=[(-1.5, -1.5), (1.5, 1.5)], cluster_std=1, random_state=0)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='r')
plt.show()


def predict_1(x: np.ndarray) -> int:
    if x[0] < 0 and x[1] < 0:
        return 0  # class A
    else:
        return 1  # class B


predictions = []
for point in X:
    predictions.append(predict_1(point))

predictions_array = np.array(predictions)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # number of correct predictions
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_true)


# something is wrong here! why is the accuracy of predict_2 lower?
def predict_2(x: np.ndarray) -> int:
    if -x[0] > x[1]:
        return 1
    else:
        return 0


predictions2 = []
for point in X:
    predictions2.append(predict_2(point))

predictions_array2 = np.array(predictions2)

print("Accuracy of predict_1: ", accuracy(y, predictions_array))
print("Accuracy of predict_2: ", accuracy(y, predictions_array2))