import numpy as np

import math

def descriptive_statistics(data):
	# Your code here

	data = np.array(data)

	mean = np.mean(data)

	median = np.median(data)
	mode = np.bincount(data).argmax() if np.issubdtype(data.dtype, np.integer) else float(np.unique(data, return_counts=True)[0][np.unique(data, return_counts=True)[1].argmax()])
	variance = np.var(data)
	std_dev = np.std(data)
	percentiles = np.percentile(data, [25, 50, 75])
	iqr = percentiles[2] - percentiles[0]
	# Create a dictionary to store the results

	stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance,4),
        "standard_deviation": np.round(std_dev,4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }
	return stats_dict