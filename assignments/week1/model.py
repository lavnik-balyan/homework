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

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fitting the model on X and y.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
        """
        # want to randomly initialize the weights and the bias
        self.b = 0
        self.w = np.random.randn(X.shape[1])
        for i in range(epochs):
            dw, db = self._compute_gradient(X, y)
            self.w = self.w - (lr * dw)
            self.b = self.b - (lr * db)

    def _compute_gradient(self, X, y):
        # calculating Xtranspose.Y
        y_pred = self.predict(X)
        diff = y_pred - y
        diff_w = np.multiply(np.divide(2, X.shape[0]), (X.T @ diff))
        diff_b = np.multiply(np.divide(2, X.shape[0]), np.sum(diff))
        return (diff_w, diff_b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return X.dot(self.w) + self.b
