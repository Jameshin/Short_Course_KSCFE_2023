###################
# Pruning
###################

import numpy as np

# Define a simple weight matrix
weights = np.array([[0.2, -0.5, 0.8],
                    [0.6, 0.1, -0.3]])

# Set the pruning threshold
threshold = 0.5

def weight_pruning(weights, threshold):
    # Create a boolean mask with True values where the absolute weight value is less than the threshold
    mask = np.abs(weights) < threshold
    
    # Set the weights to 0 where the mask is True
    pruned_weights = np.where(mask, 0, weights)
    
    return pruned_weights

# Prune the weights
pruned_weights = weight_pruning(weights, threshold)

print("Original weights:")
print(weights)

print("\nPruned weights:")
print(pruned_weights)
