import taichi as ti
import utility as utl
# 场景物体类，把bounding box和亮度场绑定在一起
@ti.data_oriented
class Shape:
    # ver_num-bounding box几何体的顶点数，tri_num-bounding box几何体的三角形数
    # id_shape-亮度场的shape（x-size, y-size, z-size）
    # org_ver_id-bounding box几何体中对应体素原点的顶点id
    # Todo: 这种绑定对亮度场发生旋转的情况是不适用的，后期再改进
    def __init__(self, ver_num, tri_num, id_shape, org_ver_id):
        self.ver = ti.Vector.field(3, ti.f32, shape=ver_num)
        self.tri = ti.Vector.field(3, ti.i32, shape=tri_num)
        self.minmax = ti.Vector.field(2, ti.f32, shape=3)
        self.id0 = ti.field(ti.f32, shape=id_shape)
        self.id1 = ti.Vector.field(3, ti.f32, shape=id_shape)
        self.org_ver_id = org_ver_id
        self.step = None
        # 最长对角线路径上的采样次数
        self.sample_num = 50

    @ti.kernel
    def minmax_coordinate(self):
        # 先进行初始化
        for _i in ti.ndrange(self.minmax.shape[0]):
            self.minmax[_i][0] = utl.infinite
            self.minmax[_i][1] = utl.infinitesimal

        for _j in ti.ndrange(self.ver.shape[0]):
            # x轴
            self.minmax[0][0] = min(self.ver[_j].x, self.minmax[0][0]) # x-min
            self.minmax[0][1] = max(self.ver[_j].x, self.minmax[0][1]) # x-max
            # y轴
            self.minmax[1][0] = min(self.ver[_j].y, self.minmax[1][0]) # y-min
            self.minmax[1][1] = max(self.ver[_j].y, self.minmax[1][1]) # y-max
            # z轴
            self.minmax[2][0] = min(self.ver[_j].z, self.minmax[2][0]) # z-min
            self.minmax[2][1] = max(self.ver[_j].z, self.minmax[2][1]) # z-max

    # start_v_id到end_v_id的对角线长度
    @ti.kernel
    def diagonal_len(self, start_v_id: ti.i32, end_v_id: ti.i32) -> ti.f32:
        _len = self.ver[end_v_id] - self.ver[start_v_id]
        return _len.norm()

    # 初始化taichi域
    def taichi_scope_init(self, ver, tri, id0, id1, diagonal_v_ids):
        self.ver.from_numpy(ver)
        self.tri.from_numpy(tri)
        self.minmax_coordinate()
        self.step = self.diagonal_len(diagonal_v_ids[0], diagonal_v_ids[1]) / self.sample_num
        self.id0.from_numpy(id0)
        self.id1.from_numpy(id1)
    
    # 缩放
    @ti.kernel
    def scale(self, vec3: ti.ext_arr()):
        _ti_mat = ti.Matrix([[vec3[0], 0, 0], [0, vec3[1], 0], [0, 0, vec3[2]]])
        
        for i in ti.ndrange(self.ver.shape[0]):
            self.ver[i] = _ti_mat@self.ver[i]
        # 更新bounding box的边界范围
        self.minmax_coordinate()

    # 平移
    @ti.kernel
    def translate(self, vec3: ti.ext_arr()):
        _vec3 = ti.Vector([vec3[0], vec3[1], vec3[2]])
        
        for i in ti.ndrange(self.ver.shape[0]):
            self.ver[i] = self.ver[i] + _vec3
        # 更新bounding box的边界范围
        self.minmax_coordinate()
       
    #########################################【坐标转换】#####################################
    # 从世界坐标转换到局部坐标，只考虑平移，不考虑旋转情况
    @ti.func
    def world2local_coordinate(self, ver):
        _org_ver = self.ver[self.org_ver_id]
        return ver - _org_ver
    # 局部坐标转换到体素坐标
    @ti.func
    def local2voxel_coordinate(self, local_p):
        _x_len = ti.abs(self.minmax[0][1]-self.minmax[0][0])
        _y_len = ti.abs(self.minmax[1][1]-self.minmax[1][0])
        _z_len = ti.abs(self.minmax[2][1]-self.minmax[2][0])
        _normalize_p = local_p / ti.Vector([_x_len, _y_len, _z_len])
        # p点所在体素位置，浮点数
        _voxel_p = _normalize_p * (ti.Vector(self.id0.shape)-1)
        return _voxel_p
    # 世界坐标转换到体素坐标
    @ti.func
    def world2voxel_coordiante(self, ver):
        _local_p = self.world2local_coordinate(ver)
        _voxel_p = self.local2voxel_coordinate(_local_p)
        return _voxel_p
    #######################################################################################

    # 获取参与介质内部current_p位置(世界坐标)处的亮度
    @ti.func
    def get_intensity(self, current_p):
        _voxel_p = self.world2voxel_coordiante(current_p)
        _voxel_p_index = ti.cast(_voxel_p, ti.i32)
        return self.id0[_voxel_p_index], self.id1[_voxel_p_index]
