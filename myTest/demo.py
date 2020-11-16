import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

mat = np.array([[8,-3, 2], [4, 11, -1], [6, 3, 12]])
a = np.zeros(shape=3)
b = np.array([20, 30, 36])
for iter in range(5):
    for i in range(3):
        temp = b[i]
        for j in range(3):
            if i != j:
                temp -= mat[i, j]*a[j]
        a[i] = temp/mat[i, i]
print(a)
