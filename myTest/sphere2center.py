import taichi as ti
import numpy as np
import math

# CUDA 在生成随机数时比 OpenGL 慢很多
ti.init(arch=ti.cpu)

# 画布大小
width = 512
height = 512
# 采样步长
sampling_step = 5
# 定义三角形
ver = np.array([(1.5878, 0.0, -2.0), (0.0, 0.0, -2.0), (0.0, 1.5, -2.0)], dtype=np.float32)
ver_index = np.array([(0, 1, 2)])
ver_4 = np.insert(ver, 3, np.array([1, 1, 1]), axis=1)

# 定义视锥参数
cone = {"l": -1.0, "r": 1.0, "t": 1.0, "b": -1.0, "n": -1.0, "f": -10.0}
cone_mat = np.array([(2*cone["n"]/(cone["r"]-cone["l"]), 0, -(cone["r"]+cone["l"])/(cone["r"]-cone["l"]), 0.),
                     (0., 2*cone["n"]/(cone["t"]-cone["b"]), -(cone["t"]+cone["b"])/(cone["t"]-cone["b"]), 0.),
                     (0., 0., (cone["f"]+cone["n"])/(cone["f"]-cone["n"]), -2*cone["f"]*cone["n"]/(cone["f"]-cone["n"])),
                     (0., 0., 1., 0.)
                     ], dtype=np.float32)
# 相机位置
cam_org = np.array([0, 0, 0])

# 定义taichi变量
center = ti.field(ti.f32, shape=3, needs_grad=True)
pixels = ti.field(ti.f32, shape=(width, height))
target_center = ti.field(ti.f32, shape=3)
L = ti.field(ti.f32, shape=(), needs_grad=True)
# 透视变换矩阵(相机坐标系->图像坐标系)
psp_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=())
# 图像坐标系->像素坐标系转换矩阵, (u, v, 1) = xy2uv_mat@xy
xy2uv_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
# 像素坐标系->图像坐标系转换矩阵，(x, y, 1) = uv2xy_mat@uv
uv2xy_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())
# 定义三角形
vertex_3 = ti.Vector.field(3, ti.f32, shape=3)
# 齐次坐标时使用
vertex_4 = ti.Vector.field(4, ti.f32, shape=3)
# 三角形顶点索引
# ver_index = ti.field(ti.f32, shape=(3, 1))

# 算法来源：https://www.cnblogs.com/graphics/archive/2010/08/09/1795348.html
# orig 射线起点
# dir 射线方向
# ver 三角形顶点位置
@ti.func
def intersect_triangle(orig, dir, ver):
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
                # print(dir_t, u, v)

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
        self.left_down_corner = [cone['l'], cone['b'], cone['n']]
        self.width = cone['r']-cone['l']
        self.height = cone['t']-cone['b']
        self.step_x = self.width / width
        self.step_y = self.height / height

    @ti.func
    def get_ray(self, _x, _y):
        origin = ti.Vector(self.origin)
        direction = ti.Vector([self.left_down_corner[0] + _x*self.step_x, self.left_down_corner[1] + _y*self.step_y, \
                self.left_down_corner[2]]) - ti.Vector(self.origin)
        return Ray(origin, direction)

@ti.func
def sampling_gradient(pix0, pix1):
    print("starting sampling...")
    # print(v0, v1)
    # norm = (v1-v0).normalized()
    p0 = ti.cast(pix0, ti.f32)
    p1 = ti.cast(pix1, ti.f32)
    p01 = p1-p0
    norm = p01.normalized()
    p01_len = p01.norm()
    pt = norm*0
    pt_len = pt.norm()

    while pt_len <= p01_len:
        pt += norm * sampling_step
        pt_len = pt.norm()
        if pt_len <= p01_len:
            pt_temp = ti.cast(p0+pt, ti.int32)
            if pixels[pt[0], pt[1]] == 0:
                pixels[pt_temp[0], pt_temp[1]] = 1


# 透视投影
# v: 为世界坐标系中的一个三维点位置，这里的v用齐次坐标表示，大小为4*1[x_w, y_w, z_w, 1]，其中psp_mat是投影矩阵
# psp_v: 为v投影到归一化的图像坐标系中的一个点，也用齐次坐标表示，大小为3*1[x_o, y_o, 1]
@ti.func
def pers_project(v):
    psp_v = psp_mat@v
    psp_v /= psp_v[3] # 归一化后的图像坐标
    return psp_v

# 图像坐标->像素坐标转换
# psp_v: 为图像坐标系中的一个点，用齐次坐标表示，大小为3*1[x_o, y_o, 1]
# uv: 为psp_v投影到像素坐标系中的一个点，也用齐次坐标表示，大小为3*1[u, v, 1]
@ti.func
def xy2uv(psp_v):
    uv = xy2uv_mat @ ti.Vector([psp_v[0], psp_v[1], 1])  # 像素坐标
    return ti.cast(uv, ti.int32)

@ti.func
def v2uv(v):
    psp_v = pers_project(v)
    return xy2uv(psp_v)

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

    # 对三角形的边进行采样
    vi = ti.Vector(ver_index)
    for i in ti.static(range(vi.n)):
        v0 = vertex_4[vi[i, 0]]
        v1 = vertex_4[vi[i, 1]]
        v2 = vertex_4[vi[i, 2]]

        pix0 = v2uv(v0)
        pix1 = v2uv(v1)
        pix2 = v2uv(v2)
        sampling_gradient(pix0, pix1)
        sampling_gradient(pix1, pix2)
        sampling_gradient(pix2, pix0)

    # for i in ti.static(range(vi.n)):
    #     print(vi[i, 0], vi[i, 1])

    # for i in range(vertex_4.shape[0]):
    #     psp_v = psp_mat@vertex_4[i]
    #     psp_v /= psp_v[3] # 归一化后的图像坐标
    #     uv = uv_mat@ti.Vector([psp_v[0], psp_v[1], 1]) # 像素坐标
    #     pixels[uv[0], uv[1]] = 1
        # print(psp_v)
        # sampling_gradient(1.0, 1.0)

    # print(psp_mat)
    # for i, j in ti.ndrange(height, width):
    #     _ray = camera.get_ray(i, j)
    #     is_hit, t, u, v = intersectTraingle(_ray.origin, _ray.direction, vertex_3)
        # is_hit = sphere.hit(_ray, ti.Vector(sphere.center))
        # if is_hit:
        #     pixels[i, j] = 1
        # else:
        #     pixels[i, j] = 0



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
camera = Camera(cam_org)
# 圆球位置
sphere = Sphere(_center, 1)
# 图像坐标系->像素坐标系转换参数
dx = (cone['r']-cone['l'])/width
dy = (cone['t']-cone['b'])/height
uo = np.around((cone['r']-cone['l'])/(2*dx))
vo = np.around((cone['t']-cone['b'])/(2*dy))
xy2uv_mat_np = np.array([(1 / dx, 0, uo), (0, 1 / dy, vo), (0, 0, 1)], dtype=np.float32)
uv2xy_mat_np = np.array([(dx, 0.0, -uo*dx), (0.0, dy, -vo*dy), (0.0, 0.0, 1)], dtype=np.float32)
# 变量赋值
center.from_numpy(np.asarray(_center))
target_center.from_numpy(np.asarray(_tar_center))
vertex_3.from_numpy(ver) # 网格顶点
vertex_4.from_numpy(ver_4) # 网格顶点-供其次坐标使用
psp_mat.from_numpy(cone_mat) # 透视投影矩阵
xy2uv_mat.from_numpy(xy2uv_mat_np) # 图像坐标->像素坐标 转换矩阵
uv2xy_mat.from_numpy(uv2xy_mat_np) # 像素坐标->图像坐标 转换矩阵
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()
    rendering()
    # exit()
    with ti.Tape(loss=L):
        reduce()
    gradient_descent()
    gui.set_image(pixels.to_numpy())
    gui.show()




