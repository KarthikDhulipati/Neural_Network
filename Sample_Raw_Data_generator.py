import numpy as np
import matplotlib.pyplot as plt


def create_data(points, classes):
    X = np.zeros((points * classes, 2))  # Shape should be (points * classes, 2) for 2D data
    y = np.zeros(points * classes, dtype='uint8')  # 1D array for labels
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]  # Create the spiral data
        y[ix] = class_number  # Assign class labels
    return X, y


# Generate data
X, y = create_data(100, 3)

# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.show()
