import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

x = 10.0
y = 0.01*np.square(x)+3
tar_y = 10
loss = 0.0

for i in range(200):
    gradient = 8*x
    x -= 0.2*gradient
    y = 4*np.square(x) - 92
    print(x)