import numpy as np
from pymcdm.weights.subjective import AHP

# define the matrix of pairwise criteria preferences, example for six criteria given below
example_matrix = np.array(
    [
        [1, 1, 1, 1/3, 1/3, 1/3],
        [1, 1, 1, 1/3, 1/3, 1/3],
        [1, 1, 1, 1/3, 1/3, 1/3],
        [3, 3, 3, 1, 1, 1],
        [3, 3, 3, 1, 1, 1],
        [3, 3, 3, 1, 1, 1],
    ]
)

# calculate weights
ahp = AHP(matrix=example_matrix)
weights = ahp()

# print weights for use in experiment configuration
print(weights)
