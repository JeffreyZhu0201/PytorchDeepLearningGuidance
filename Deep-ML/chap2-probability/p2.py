import numpy as np

import math

def binomial_probability(n, k, p):
	"""
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
	# Your code here

	probability = (math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k)))

	return round(probability, 5)