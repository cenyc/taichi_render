import taichi as ti
import numpy as np
import math
ti.init(arch=ti.cpu)

h = 512
w = 512

I = ti.field(ti.f32, shape=(h, w))

@ti.kernel
def test():
    for i, j in ti.ndrange(h, w):
        I[i, j] = 1.0

test()

gui = ti.GUI("RTE_DR")
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()

    gui.set_image(I.to_numpy())
    gui.show()