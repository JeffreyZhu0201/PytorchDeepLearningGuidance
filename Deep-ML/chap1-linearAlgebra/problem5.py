
import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])


def cosine_similarity(v1, v2):
	# Implement your code here
	# Calculate the cosine similarity between two vectors

    return round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),3)


print(cosine_similarity(v1, v2))