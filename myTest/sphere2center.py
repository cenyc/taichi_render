import taichi as ti
import numpy as np
import math

# CUDA 在生成随机数时比 OpenGL 慢很多
ti.init(arch=ti.gpu)

# 画布大小
width = 512
height = 512

# a = ti.Vector
center = ti.field(ti.f32, shape=3, needs_grad=True)
pixels = ti.field(ti.f32, shape=(width, height))
target_center = ti.field(ti.f32, shape=3)
L = ti.field(ti.f32, shape=(), needs_grad=True)

# 定义三角形
vertex = ti.field(ti.f32, shape=(3,3))
ver_index = ti.field(ti.f32, shape=(3,1))

vertex = [[0.5, 0, 2], [0, 0, 2], [0, 0.5, 2]]
ver_index = [[0, 1, 2]]

@ti.func
def intersectTraingle(vertex, ver_index):
    a = vertex[0]-vertex[1]
    print(a)

@ti.kernel
def test():
    intersectTraingle(vertex, ver_index)

test()

exit()


@ti.data_oriented
class Ray:
    def __init__(self, org, dir):
        self.origin = org
        self.direction = dir

@ti.data_oriented
class Sphere:
    def __init__(self, cent, r):
        self.center = np.asarray(cent)
        self.radius = r

    @ti.func
    def hit(self, _ray, cc):
        oc = _ray.origin - cc

        a = _ray.direction.dot(_ray.direction)
        b = oc.dot(_ray.direction)
        c = oc.dot(oc) - self.radius*self.radius
        discriminant = b*b - a*c
        is_hit = 0
        if discriminant > 0 :
            is_hit = 1
        return is_hit


@ti.data_oriented
class Camera:
    def __init__(self, org):
        self.origin = org
        self.left_down_corner = [-1, -1, 0]
        self.width = 2
        self.height = 2
        self.step_x = self.width / width
        self.step_y = self.height / height

    @ti.func
    def get_ray(self, _x, _y):
        origin = ti.Vector(self.origin)
        direction = ti.Vector([self.left_down_corner[0] + _x*self.step_x, self.left_down_corner[1] + _y*self.step_y, \
                self.left_down_corner[2]]) - ti.Vector(self.origin)
        return Ray(origin, direction)

@ti.kernel
def rendering():
    # 渲染球体
    # sphere.__init__([center[0], center[1], center[2]], 1)
    # for i, j in ti.ndrange(height, width):
    #     _ray = camera.get_ray(i, j)
    #     is_hit = sphere.hit(_ray, ti.Vector(sphere.center))
    #     if is_hit:
    #         pixels[i, j] = 0
    #     else:
    #         pixels[i, j] = 1
    # print(sphere.center[0])

    # 渲染三角形
    print("rending triangle...")




@ti.kernel
def reduce():
    temp = ti.Vector([center[0]-target_center[0], center[1]-target_center[1], center[2]-target_center[2]])
    L[None] += 0.05 * temp.norm_sqr()

@ti.kernel
def gradient_descent():
        center[0] -= center.grad[0] * 0.05
        center[1] -= center.grad[1] * 0.05
        center[2] -= center.grad[2] * 0.05


gui = ti.GUI('SDF 2D')
_center = [-1, -1, 2]
_tar_center = [0, 0, 2]
camera = Camera([0, 0, -1])
sphere = Sphere(_center, 1)

vertex = [[0.5, 0, 2], [0, 0, 2], [0, 0.5, 2]]
ver_index = [[0, 1, 2]]
center.from_numpy(np.asarray(_center))
target_center.from_numpy(np.asarray(_tar_center))


while True:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()
    rendering()
    with ti.Tape(loss=L):
        reduce()
    gradient_descent()
    gui.set_image(pixels.to_numpy())
    gui.show()




