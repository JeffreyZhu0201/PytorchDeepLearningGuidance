import numpy as np
import math
import time
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    
    
    m, n = X.shape
    y = np.array([1, 2, 3]).reshape(-1, 1)  # Reshape y to be a column vector
    theta = np.zeros((n, 1))    # 参数矩阵(n,1)
    for _ in range(iterations):
        predictions = X @ theta# (m,1)
        errors = predictions - y# (m,1)
        gradient = (X.T @ errors) / m   # (n,1)
        theta -= alpha * gradient   # 更新参数
    
    theta = np.round(theta,4)
    # print(f"Training time: {end_time - start_time} seconds")
    return theta

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3]).reshape(-1, 1)  # Reshape y to be a column vector
alpha = 0.01
iterations = 1000
theta = linear_regression_gradient_descent(X, y, alpha, iterations)
print(theta)