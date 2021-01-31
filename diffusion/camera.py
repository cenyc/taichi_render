import taichi as ti
import numpy as np
import math

from ray import Ray

cone = {"l": -1.0, "r": 1.0, "t": 1.0, "b": -1.0, "n": -1.0, "f": -10.0}
cone_mat = np.array([(2*cone["n"]/(cone["r"]-cone["l"]), 0, -(cone["r"]+cone["l"])/(cone["r"]-cone["l"]), 0.),
                     (0., 2*cone["n"]/(cone["t"]-cone["b"]), -(cone["t"]+cone["b"])/(cone["t"]-cone["b"]), 0.),
                     (0., 0., (cone["f"]+cone["n"])/(cone["f"]-cone["n"]), -2*cone["f"]*cone["n"]/(cone["f"]-cone["n"])),
                     (0., 0., 1., 0.)
                     ], dtype=np.float32)
# 画布大小
width = 512
height = 512

@ti.data_oriented
class Camera:
    def __init__(self):
        self.origin = ti.Vector.field(3, ti.f32, shape=())
        self.translation = ti.Vector.field(3, ti.f32, shape=(), needs_grad=True)
        # self.horizontal = dir / np.linalg.norm(dir)
        self.left_down_corner = ti.Vector.field(3, ti.f32, shape=())
        # self.left_down_corner = ti.Vector([cone['l'], cone['b'], cone['n']])
        self.width = cone['r']-cone['l']
        self.height = cone['t']-cone['b']
        self.resolution_w = width
        self.resolution_h = height
        self.step_x = self.width / self.resolution_w
        self.step_y = self.height / self.resolution_h
        

    @ti.kernel
    def taichi_scope_init(self):
        self.left_down_corner = ti.Vector([cone['l'], cone['b'], cone['n']])

    # 从像素点_x, _y获取发出的一条光线
    @ti.func
    def get_ray(self, _x, _y):
        # _origin = ti.Vector(self.origin)
        _direction = self.left_down_corner + ti.Vector([_x*self.step_x, _y*self.step_y, 0.0]) - self.origin
        _direction.normalized()
        return Ray(self.origin, _direction)

    # 对相机位置进行平移操作
    @ti.kernel
    def translate(self, vec3: ti.ext_arr()):
        self.translation = ti.Vector([vec3[0], vec3[1], vec3[2]])
        self.left_down_corner += self.translation
        self.origin += self.translation
