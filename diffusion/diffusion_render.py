import taichi as ti
import numpy as np
import math
import os
# CUDA 在生成随机数时比 OpenGL 慢很多
ti.init(arch=ti.cpu)

# *********************************************************************************************
# --------------------------------------------参数预定义-----------------------------------------
# *********************************************************************************************
# 画布大小
width = 512
height = 512
# 新定义 start
ver = []
tri = []
# 定义视锥参数
cone = {"l": -1.0, "r": 1.0, "t": 1.0, "b": -1.0, "n": -1.0, "f": -10.0}
cone_mat = np.array([(2*cone["n"]/(cone["r"]-cone["l"]), 0, -(cone["r"]+cone["l"])/(cone["r"]-cone["l"]), 0.),
                     (0., 2*cone["n"]/(cone["t"]-cone["b"]), -(cone["t"]+cone["b"])/(cone["t"]-cone["b"]), 0.),
                     (0., 0., (cone["f"]+cone["n"])/(cone["f"]-cone["n"]), -2*cone["f"]*cone["n"]/(cone["f"]-cone["n"])),
                     (0., 0., 1., 0.)
                     ], dtype=np.float32)
dx = (cone['r']-cone['l'])/width
dy = (cone['t']-cone['b'])/height
uo = np.around((cone['r']-cone['l'])/(2*dx))
vo = np.around((cone['t']-cone['b'])/(2*dy))
xy2uv_mat_np = np.array([(1 / dx, 0, uo), (0, 1 / dy, vo), (0, 0, 1)], dtype=np.float32)
uv2xy_mat_np = np.array([(dx, 0.0, -uo*dx), (0.0, dy, -vo*dy), (0.0, 0.0, 1)], dtype=np.float32)

# 定义taichi变量
# 渲染结果图像
img = ti.field(ti.f32, shape=(height, width))

# *********************************************************************************************
# -------------------------------------------方法定义-------------------------------------------
# *********************************************************************************************
# 读取off文件
def readOFF(file_path):
    _ver = []
    _tri = []
    _count = 0
    with open(file_path, 'r') as _f:
        for _line in _f.readlines():
            _line = _line.strip('\n').split(' ')
            if len(_line) == 3 and _count != 1:
                _ver.append(_line)
            elif len(_line) == 4:
                _tri.append(_line)
            _count += 1
    return _ver, _tri

@ti.data_oriented
class Ray:
    def __init__(self, org, dir):
        self.origin = org
        self.direction = dir

@ti.data_oriented
class Camera:
    def __init__(self, org):
        self.origin = org
        self.left_down_corner = [cone['l'], cone['b'], cone['n']]
        self.width = cone['r']-cone['l']
        self.height = cone['t']-cone['b']
        self.step_x = self.width / width
        self.step_y = self.height / height
    # 从像素点_x, _y获取发出的一条光线
    @ti.func
    def get_ray(self, _x, _y):
        _origin = ti.Vector(self.origin)
        _direction = ti.Vector([self.left_down_corner[0] + _x*self.step_x, self.left_down_corner[1] + _y*self.step_y, \
                self.left_down_corner[2]]) - ti.Vector(self.origin)
        return Ray(_origin, _direction)

# orig 射线起点
# dir 射线方向
# ver 三角形顶点位置
# 算法来源：https://www.cnblogs.com/graphics/archive/2010/08/09/1795348.html
@ti.func
def intersect_triangle(orig, dir, ver):
    _is_hit = 1; _dir_t = 0.0; _u = 0.0; _v = 0.0
    # E1
    _e1 = ver[1] - ver[0]
    # E2
    _e2 = ver[2] - ver[0]
    # P
    _p = dir.cross(_e2)
    # determinant
    _det = _e1.dot(_p)
    _t = ti.Vector([0.0, 0.0, 0.0])
    if _det > 0:
        _t = orig - ver[0]
    else:
        _t = ver[0] - orig
        _det = -_det

    if _det < 0.0001:
        _is_hit = 0
    else:
        _u = _t.dot(_p)

        if _u < 0.0 or _u > _det:
            _is_hit = 0
        else:
            _q = _t.cross(_e1)
            _v = dir.dot(_q)
            if _v < 0.0 or _u+_v >_det:
                _is_hit = 0
            else:
                _dir_t = _e2.dot(_q)
                _fInvDet = 1.0/_det
                _dir_t *= _fInvDet
                _u *= _fInvDet
                _v *= _fInvDet
                # print(dir_t, u, v)

    return _is_hit, _dir_t, _u, _v


@ti.func
def get_img(ver, img):
    for i, j in ti.ndrange(height, width):
        _ray = camera.get_ray(i, j)
        _is_hit, t, u, v = intersect_triangle(_ray.origin, _ray.direction, ver)
        if _is_hit:
            img[i, j] = 1
        else:
            img[i, j] = 0

# *********************************************************************************************
# ---------------------------------------------程序运行-----------------------------------------
# *********************************************************************************************
# 相机位置
cam_org = np.array([0, 0, 0])
camera = Camera(cam_org)
ver, tri = readOFF('./diffusion/cube.off')
# 定义taichi变量
ti_ver3 = ti.Vector.field(3, ti.f32, shape=len(ver))
gui = ti.GUI('diffusion')
print(len(ver))




exit(0)

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------楚------------------------------------------------河---------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------




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

# 像素坐标->图像坐标转换
# uv: 像素坐标，使用其次坐标表示，大小为3*1[u, v, 1]
# xy: 图像坐标，使用齐次坐标表述，大小为3*1[x, y, 1]
@ti.func
def uv2xy(uv):
    return uv2xy_mat @ uv

@ti.func
def v2uv(v):
    psp_v = pers_project(v)
    return xy2uv(psp_v)



@ti.kernel
def get_target_img():
    get_img(tar_ver3, target_img)

@ti.kernel
def get_source_img():
    get_img(ver3, source_img)
    for i, j in ti.ndrange(height, width):
        # diff_img[i, j] = ti.abs(target_img[i, j] - source_img[i, j])
        diff_img[i, j] = source_img[i, j] - target_img[i, j]
        # if diff_img[i, j] != 0:
        #     diff_img[i, j] = 1



@ti.kernel
def rendering():
    # 对三角形的边进行采样
    vi = ti.Vector(ver_index)
    for i in ti.static(range(vi.n)):
        v0 = ver3[vi[i, 0]]
        v1 = ver3[vi[i, 1]]
        v2 = ver3[vi[i, 2]]

        # pix0 = v2uv(v0)
        # pix1 = v2uv(v1)
        # pix2 = v2uv(v2)

        # sampling_gradient(v0, v1, 0, 0)
        # sampling_gradient(v1, v2, 1, type=0)
        # sampling_gradient(v2, v0, 2, type=0)

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

@ti.func
def clear_img():
    # clear gradient_img
    for i, j, k in ti.ndrange(height, width, 3):
        gradient_img[i, j, k] = ti.Vector([0.0, 0.0])
    # clear debug_img
    for i, j in ti.ndrange(height, width):
        debug_img[i, j] = 0.0

@ti.kernel
def postprocessing():
    clear_img()

# 开始进行可微渲染计算
# ----------------------------------------------------
gui = ti.GUI('DR')
_center = [-1, -1, 2]
_tar_center = [0, 0, 2]
# 相机位置
camera = Camera(cam_org)

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
ver3.from_numpy(ver3_np) # 网格顶点
tar_ver3.from_numpy(tar_ver3_np)
ver4.from_numpy(ver4_np) # 网格顶点-供其次坐标使用
tar_ver4.from_numpy(tar_ver4_np)
psp_mat.from_numpy(cone_mat) # 透视投影矩阵
xy2uv_mat.from_numpy(xy2uv_mat_np) # 图像坐标->像素坐标 转换矩阵
uv2xy_mat.from_numpy(uv2xy_mat_np) # 像素坐标->图像坐标 转换矩阵

# 获取目标图像
get_target_img()
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()
    get_source_img()
    gui.set_image(diff_img.to_numpy())
    # gui.set_image(I)
    gui.show()
    postprocessing()




