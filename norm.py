import numpy as np

def get_data_stats():
    min_values = np.array([0.000] * 23)  
    max_values = np.array([1.000] * 23) 
    min_values[3] = -1.000  
    max_values[3] = 1.000   
    min_values[11] = -1.000  
    max_values[11] = 1.000   
    min_values[16:] = np.array([-0.28, -0.78, -1.19, 0.13, -0.15, 0.14, -2.79])
    max_values[16:] = np.array([0.66,  0.20,  0.17,  1.67, 1.06,  1.68, -0.71])

    stats = {
        'min': min_values,
        'max': max_values
    }
    return stats

def generate_test_data(n):
    stats = get_data_stats()
    
    min_values = stats['min']
    max_values = stats['max']
    
    data = np.zeros((n, 23))
    
    data[:, [3, 11]] = np.random.uniform(-1, 1, (n, 2))
    
    for i in range(16):
        if i not in [3, 11]:
            data[:, i] = np.random.uniform(0, 1, n)
    
    last_7_start = 16
    last_7_end = 23
    for i in range(last_7_start, last_7_end):
        data[:, i] = np.random.uniform(min_values[i], max_values[i], n)
    
    return data

def normalize_data(data, stats):
    ndata = np.zeros_like(data)
    
    # Normalize first 16 dimensions
    ndata[:, [3, 11]] = data[:, [3, 11]]
    
    # Normalize the other 14 dimensions to [-1, 1]
    for i in range(16):
        if i not in [3, 11]:  # dimensions 4 and 12
            ndata[:, i] = (data[:, i] * 2) - 1  # Mapping from [0, 1] to [-1, 1]
    
    # Normalize the last 7 dimensions
    last_7_start = 16
    last_7_end = 23
    
    for i in range(last_7_start, last_7_end):
        range_ = stats['max'][i] - stats['min'][i]
        ndata[:, i] = 2 * (data[:, i] - stats['min'][i]) / range_ - 1  # Mapping from [min, max] to [-1, 1]
    
    return ndata

def unnormalize_data(ndata, stats):
    data = np.zeros_like(ndata)
    
    # Unnormalize first 16 dimensions
    data[:, [3, 11]] = ndata[:, [3, 11]]
    
    # Unnormalize the other 14 dimensions from [-1, 1] to [0, 1]
    for i in range(16):
        if i not in [3, 11]:  # dimensions 4 and 12
            data[:, i] = (ndata[:, i] + 1) / 2  # Mapping from [-1, 1] to [0, 1]
    
    # Unnormalize the last 7 dimensions
    last_7_start = 16
    last_7_end = 23
    
    for i in range(last_7_start, last_7_end):
        range_ = stats['max'][i] - stats['min'][i]
        data[:, i] = (ndata[:, i] + 1) / 2 * range_ + stats['min'][i]  # Restore to original range
    
    return data

# Generate test data with 3 samples
test_data = generate_test_data(3)

# Set print options to show 2 decimal places
np.set_printoptions(precision=2, suppress=True)

# Print the original test data
print("Original Test Data:")
print(test_data)

# Normalize the test data
stats = get_data_stats()
normalized_data = normalize_data(test_data, stats)

# Print the normalized data
print("\nNormalized Data:")
print(normalized_data)

# Unnormalize the normalized data
unnormalized_data = unnormalize_data(normalized_data, stats)

# Print the unnormalized data
print("\nUnnormalized Data:")
print(unnormalized_data)