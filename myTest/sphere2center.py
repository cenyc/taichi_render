import taichi as ti
import numpy as np
import math

# CUDA 在生成随机数时比 OpenGL 慢很多
ti.init(arch=ti.gpu)

# 画布大小
width = 512
height = 512

# 定义三角形
ver = np.array([(1.5, 0.0, -2.0), (0.0, 0.0, -2.0), (0.0, 1.5, -2.0)], dtype=np.float32)
ver_index = np.array([0, 1, 2])

# 定义视锥参数
cone = {"l": -1, "r": 1, "t": 1, "b": -1, "n": -1, "f": 10}

# a = ti.Vector
center = ti.field(ti.f32, shape=3, needs_grad=True)
pixels = ti.field(ti.f32, shape=(width, height))
target_center = ti.field(ti.f32, shape=3)
L = ti.field(ti.f32, shape=(), needs_grad=True)

# 定义三角形
vertex = ti.Vector.field(3, ti.f32, shape=3)
ver_index = ti.field(ti.f32, shape=(3,1))

# 算法来源：https://www.cnblogs.com/graphics/archive/2010/08/09/1795348.html
# orig 射线起点
# dir 射线方向
# ver 三角形顶点位置
@ti.func
def intersectTraingle(orig, dir, ver):
    is_hit = 1; dir_t = 0.0; u = 0.0; v = 0.0
    # E1
    e1 = ver[1] - ver[0]
    # E2
    e2 = ver[2] - ver[0]
    # P
    p = dir.cross(e2)
    # determinant
    det = e1.dot(p)
    t = ti.Vector([0.0, 0.0, 0.0])
    if det > 0:
        t = orig - ver[0]
    else:
        t = ver[0] - orig
        det = -det

    if det < 0.0001:
        is_hit = 0
    else:
        u = t.dot(p)

        if u < 0.0 or u > det:
            is_hit = 0
        else:
            q = t.cross(e1)
            v = dir.dot(q)
            if v < 0.0 or u+v >det:
                is_hit = 0
            else:
                dir_t = e2.dot(q)
                fInvDet = 1.0/det
                dir_t *= fInvDet
                u *= fInvDet
                v *= fInvDet
                print(dir_t, u, v)

    return is_hit, dir_t, u, v


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
        self.left_down_corner = [-1, -1, -1]
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
    for i, j in ti.ndrange(height, width):
        _ray = camera.get_ray(i, j)
        is_hit, t, u, v = intersectTraingle(_ray.origin, _ray.direction, vertex)
        # is_hit = sphere.hit(_ray, ti.Vector(sphere.center))
        if is_hit:
            pixels[i, j] = 1
        else:
            pixels[i, j] = 0



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
# 相机位置
camera = Camera([0, 0, 0])
# 圆球位置
sphere = Sphere(_center, 1)
# 变量赋值
center.from_numpy(np.asarray(_center))
target_center.from_numpy(np.asarray(_tar_center))
vertex.from_numpy(ver)

while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()
    rendering()
    with ti.Tape(loss=L):
        reduce()
    gradient_descent()
    gui.set_image(pixels.to_numpy())
    gui.show()




