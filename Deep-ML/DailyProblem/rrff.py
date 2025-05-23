
import numpy as np

def rref(matrix):
	
    """
    """

    matrix = matrix.astype(float)
    m, n = matrix.shape
    pivot_row = 0

    for col in range(n):
        if pivot_row >= m:
            break

        # Find the pivot
        max_row = np.argmax(np.abs(matrix[pivot_row:m, col])) + pivot_row
        if matrix[max_row, col] == 0:
            continue

        # Swap the current row with the pivot row
        matrix[[pivot_row, max_row]] = matrix[[max_row, pivot_row]]

        # Normalize the pivot row
        matrix[pivot_row] /= matrix[pivot_row, col]

        # Eliminate the current column in other rows
        for row in range(m):
            if row != pivot_row:
                matrix[row] -= matrix[row, col] * matrix[pivot_row]

        pivot_row += 1

    return matrix