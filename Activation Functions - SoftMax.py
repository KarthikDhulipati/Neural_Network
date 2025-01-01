import math
import numpy as np
import nnfs
'''
Here's the working of the SoftMax Function:

    [ 1 ]       [ e^1 ]     [ (e^1)/(e^1+e^2+e^3) ]     [ 0.09 ]
    | 2 |   =   | e^2 | =   | (e^2)/(e^1+e^2+e^3) | =   | 0.24 |
    [ 3 ]       [ e^3 ]     [ (e^3)/(e^1+e^2+e^3) ]     [ 0.67 ]
    
    S_(i,j)  =  e^(z_(i,j) ) / (∑_(l=1)^L▒e^(Z_(i,j) ) )
'''

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# E = math.e
# # E = 2.71828182846
#
# exp_values = [] # Exponentiated values, i.e., e^x
#
# for output in layer_outputs:
#     exp_values.append(E**output)

# print("Exponentiated values : ", exp_values)

'''Now, we normalize the values in the newly created array - 'exp_values'

# Normalization process:'''

# norm_base = sum(exp_values)
# norm_values = []
#
# for value in exp_values:
#     norm_values.append(value/norm_base)

# print("Normalized Values : ", norm_values)
# print("Sum of Normalized Values : ", sum(norm_values))


'''Another Approach using numpy: '''
exp_values = np.exp(layer_outputs)

print("Exp_values: ", np.sum(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)

