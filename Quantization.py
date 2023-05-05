###################
# Quantization
###################

import numpy as np

def uniform_quantization(weights, num_bits):
    q_min, q_max = -2**(num_bits - 1), 2**(num_bits - 1) - 1
    min_val, max_val = np.min(weights), np.max(weights)
    scale = (max_val - min_val) / (q_max - q_min)
    quantized_weights = np.round((weights - min_val) / scale + q_min)
    return quantized_weights.astype(int), scale, min_val

def dequantize_weights(quantized_weights, scale, min_val, q_min):
    dequantized_weights = (quantized_weights - q_min) * scale + min_val
    return dequantized_weights

# Define a simple weight matrix
weights = np.array([[0.2, -0.5, 0.8],
                    [0.6, 0.1, -0.3]])

# Set the number of bits for quantization (e.g., 8 bits)
num_bits = 8

# Quantize the weights
integer_quantized_weights, scale, min_val = uniform_quantization(weights, num_bits)

# Calculate the quantization range minimum
q_min = -2**(num_bits - 1)

# Dequantize the weights
dequantized_weights = dequantize_weights(integer_quantized_weights, scale, min_val, q_min)

print("Original weights:")
print(weights)

print("\nInteger quantized weights:")
print(integer_quantized_weights)

print("\nDequantized weights:")
print(dequantized_weights)