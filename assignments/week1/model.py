import numpy as np

class LinearRegression:

    """
    Defining parameters
    """
    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fitting the model on X and y.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
        """
        self.w = np.linalg.solve(X.T.dot(X), X.T.dot(y))
        self.b = np.mean(y) - np.mean(X.dot(self.w))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X.dot(self.w) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000) -> None:
        """
        Fitting the model on X and y.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
        """
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        raise NotImplementedError()
