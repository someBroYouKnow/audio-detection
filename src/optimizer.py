import numpy as np


class NaiveOptimizer:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y

            loss = np.mean(error**2)
            self.loss_history.append(loss)

            dw = (2 / num_samples) * np.dot(X.T, error)
            db = (2 / num_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y = np.array([3, 5, 7, 9], dtype=np.float32)

    model = NaiveOptimizer(learning_rate=0.01, epochs=2000)
    print(type(model), model, model.weights)
    model.fit(X, y)

    print(f"Weight: {model.weights[0]:.2f}")
    print(f"Bias: {model.bias:.2f}")
    print(f"Final loss: {model.loss_history[-1]:.6f}")
