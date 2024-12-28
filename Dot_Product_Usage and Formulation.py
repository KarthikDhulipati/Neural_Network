import numpy as np

'''
We are using a dot product because a dot product has the following formula:

If  a = [1, 2, 3],
    b = [2, 3, 4], then
   
        Dot product of a & b = a.b = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
        
        => a.b = [1, 2, 3].[2, 3, 4] = (1*2) + (2*3) + (3*4)
                                     = 2 + 6 + 12
                                     = 20  
'''

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases   # USE THE WEIGHTS FIRST AND THEN THE INPUTS TO TAKE CARE OF THE LIST DIMENSION MISMATCH


print("Output = ", output)
