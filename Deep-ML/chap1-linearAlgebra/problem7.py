
import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	# Convert the input list to a NumPy array	特征值λ
	arr = np.array(matrix)

	# Calculate the eigenvalues using NumPy's linear algebra module
	eigenvalues, _ = np.linalg.eig(arr)


	return eigenvalues.tolist()