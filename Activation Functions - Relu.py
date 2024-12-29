import numpy as np

np.random.seed(0)

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

'''
This is the ReLu, i.e., Rectified Linear Activation Function's working below:
'''

for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)