
import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	inv_matrix = np.linalg.inv(matrix)
	return inv_matrix.tolist()