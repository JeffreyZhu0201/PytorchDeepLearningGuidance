import numpy as np

# def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
# 	# Your code here

# 	return np.cov(np.array(vectors), rowvar=True).tolist()
	

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here

	vectors = np.array(vectors)
	mean = np.mean(vectors, axis=1)
	cov_matrix = np.dot(((vectors - mean[:, np.newaxis])), (vectors - mean[:, np.newaxis]).T)/(len(vectors[0])-1)
	cov_matrix = np.round(cov_matrix, 1)

	return cov_matrix

print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))

# torch.cuda.is_available()
# print(torch.version.cuda)