
import taichi as ti
# 碰撞类，记录碰撞的位置及方向
@ti.data_oriented
class Interaction:
    def __init__(self, hit_p, dir):
        self.hit_p = hit_p
        self.dir = dir