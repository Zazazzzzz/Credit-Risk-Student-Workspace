probabilities = [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018]
values = [109.37, 109.19, 108.66, 107.55, 102.02, 98.10, 83.64, 51.13]

mean_value = sum(p * v for p, v in zip(probabilities, values))
variance = sum(p * ((v - mean_value) ** 2) for p, v in zip(probabilities, values))
standard_deviation = variance ** 0.5

print(f"Mean Value: {mean_value:.2f}")
print(f"Variance: {variance:.4f}")
print(f"Standard Deviation: {standard_deviation:.2f}")