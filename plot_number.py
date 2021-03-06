# -----------------------------------------------------------------------------
# Simple Neural Network for handwritten digit recognition
# 2016  Pierre-Antoine Cinquin (Github : PACinquin)
#
# Done on Python 3.5.2 |Anaconda custom (x86_64)|
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Open and read dataset
data_file = open ("mnist_test_10.csv", 'r')
data_list = data_file.readlines()
data_file.close()

# Split the data row with ','
all_values = data_list[0].split(',')

# Cut the index (first value) and make a 28x28 matrix
image_matrix = np.asfarray(all_values[1:]).reshape((28,28))

# Plot the matrix and the number appears!!
plt.imshow(image_matrix, cmap='Greys', interpolation="none")
plt.show
