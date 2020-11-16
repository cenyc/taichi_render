import taichi as ti
import numpy as np
import math
ti.init(arch=ti.cpu)

h = 512
w = 512

I = ti.field(dtype=ti.f32, shape=(h, w))
read_np = np.ndarray(dtype=np.uint8, shape=(h, w))
read_np = ti.imread('target.png')


@ti.kernel
def test():
    # for i, j in ti.ndrange(h, w):
    #     if i > 50 and i < 461:
    #         I[i, j] = 0.5
    print('test')

test()

gui = ti.GUI("RTE_DR")
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()

    gui.set_image(read_np)
    gui.show()