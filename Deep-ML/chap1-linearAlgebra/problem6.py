
import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == 'row':
		means = np.mean(matrix, axis=1)
	elif mode == 'column':
		means = np.mean(matrix, axis=0)
	return means.tolist()