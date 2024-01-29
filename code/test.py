import numpy as np


array_2d = np.array([[4, 5, 6],
                    [8, 1, 3],
                    [7, 8, 9]])

sorted_array = np.sort(array_2d, axis=0)

print(sorted_array)
