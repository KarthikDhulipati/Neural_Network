import math
import numpy

layer_outputs = [4.8, 1.21, 2.385]

E = math.e
# E = 2.71828182846

exp_values = [] # Exponentiated values, i.e., e^x

for output in layer_outputs:
    exp_values.append(E**output)

print("Exponentiated values : ", exp_values)

'''Now, we normalize the values in the newly created array - 'exp_values'

# Normalization process:'''

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print("Normalized Values : ", norm_values)
print("Sum of Normalized Values : ", sum(norm_values))
