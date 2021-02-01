import taichi as ti
import numpy as np
import math
import os
from camera import Camera
from shape import Shape
from scene import Scene
import utility as utl
# CUDA 在生成随机数时比OpenGL慢很多
ti.init(arch=ti.cpu, excepthook=True)

# *********************************************************************************************
# --------------------------------------------读取外部数据---------------------------------------
# *********************************************************************************************
# 读取模型数据与亮度场数据
ver, tri = utl.readOFF('./diffusion/data/cube.off')
id0 = np.load('./diffusion/data/id0.npy').astype(np.float32)
id1 = np.load('./diffusion//data/id1.npy').astype(np.float32)

# *********************************************************************************************
# ----------------------------------------实例化类，分配存储空间-----------------------------------
# *********************************************************************************************
# 创建相机
camera = Camera()
# 创建物体，预分配顶点存储空间与三角形面片索引存储空间
# def __init__(self, ver_num, tri_num, id_shape, org_ver_id):
shape = Shape(len(ver), len(tri), id0.shape, 0)
# 创建场景，关联相机与物体
scene = Scene(camera, shape)
ti_img = ti.field(ti.f32, shape=(camera.resolution_h, camera.resolution_w))
ti_img_tar = ti.field(ti.f32, shape=(camera.resolution_h, camera.resolution_w))

# *********************************************************************************************
# ----------------------------------------在taichi域中赋值---------------------------------------
# *********************************************************************************************
# taichi类初始化
camera.taichi_scope_init()
# def taichi_scope_init(self, ver, tri, id0, id1, diagonal_v_ids):
shape.taichi_scope_init(ver, tri, id0, id1, [0, 6])
# 将shape缩放和平移到合适的位置（调整到相机中心）
shape.scale(np.array([0.2, 0.3, 0.2], dtype=np.float32)*3)
shape.translate(np.array([-0.25, -0.25, -2], dtype=np.float32))

# *********************************************************************************************
# -------------------------------------------程序运行-------------------------------------------
# *********************************************************************************************
# 渲染目标图像
scene.render()
utl.copy(scene.img, ti_img_tar)

# 对相机视角进行平移，渲染源图像
scene.cam.translate(np.array([-0.5, -0.5, 0]).astype(np.float32))
scene.render()
utl.copy(scene.img, ti_img)

gui = ti.GUI('diffusion', (camera.resolution_w, camera.resolution_h))
while gui.running:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            exit()
    gui.set_image(ti_img_tar.to_numpy())
    gui.show()