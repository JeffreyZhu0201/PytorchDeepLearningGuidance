import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
	# Your code here
	
    m,n = X.shape
    y = np.reshape(y, (-1, 1))  # Reshape y to be a column vector

    weights = np.reshape(weights, (-1, 1))  # Reshape weights to be a column vector (n, 1)

    # batch_num = m // batch_size

    for _ in range(n_iterations):


        if method == 'batch':
        
            # start = i * n
            # end = start + n
            # X_batch = X[start:end]  # (batch_size, n)
            # y_batch = y[start:end]  # (batch_size, 1)
            predictions = X @ weights # (batch_size, 1)
            errors = (predictions - y)  #(batch_size, 1)
            gradient = 2*(X.T @ errors) / m    # (n, 1)
            weights -= learning_rate * gradient
    # weights = np.round(weights, 4)

        elif method == 'stochastic':
            for i in range(m):
                predictions = X[i] @ weights    # (1, 1)
                errors = predictions - y[i]     # (1, 1)
                gradient = 2 * X[i][:, np.newaxis] * errors
                weights -= learning_rate * gradient
        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                predictions = X_batch @ weights
                errors = (predictions - y_batch)
                gradient = 2*(X_batch.T @ errors) / batch_size
                weights -= learning_rate * gradient

    return weights.reshape(-1).tolist()  # Reshape back to 1D list

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]) 
y = np.array([2, 3, 4, 5]) 
weights = np.zeros(X.shape[1]) 
learning_rate = 0.01 
n_iterations = 100 # Test Stochastic Gradient Descent 
output = gradient_descent(X, y, weights, learning_rate, n_iterations, method='stochastic') 
print(output)