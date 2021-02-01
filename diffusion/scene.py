import taichi as ti
import utility as utl
from interaction import Interaction
@ti.data_oriented
class Scene:
    def __init__(self, cam, shape):
        self.cam = cam
        self.shape = shape
        self.img = ti.field(ti.f32, shape=(cam.resolution_h, cam.resolution_w))
        self.ver_3_tmp = ti.Vector.field(3, ti.f32, shape=3)
    
    @ti.func
    def ray_marching(self, inter_min, inter_max):
        # 将要采样的路径长度
        _path_len = (inter_max.hit_p - inter_min.hit_p).norm()
        # 当前采样点
        _current_p = inter_min.hit_p
        # 累积前进路径长度
        _cumulative_path_len = 0.0
        _intensity = 0.0
        while _cumulative_path_len <= _path_len:
            _id0, _id1 = self.shape.get_intensity(_current_p)
            # Todo
            _intensity += -inter_min.dir.dot(_id1)
            # print(_id0, -inter_min.dir.dot(_id1))
            _intensity += _id0
            _cumulative_path_len += self.shape.step

        return _intensity

    @ti.kernel
    def render(self):
        utl.zero(self.img)
        for _i, _j in ti.ndrange(self.cam.resolution_h, self.cam.resolution_w):
            _ray = self.cam.get_ray(_i, _j)
            _dir_t_max = utl.infinitesimal
            _dir_t_min = utl.infinite

            # 求光线从参与介质中穿过的最近_dir_t_min与最远_dir_t_max
            for _k in ti.ndrange(self.shape.tri.shape[0]):
                _tri = self.shape.tri[_k]
                self.ver_3_tmp[0] = self.shape.ver[_tri.x]
                self.ver_3_tmp[1] = self.shape.ver[_tri.y]
                self.ver_3_tmp[2] = self.shape.ver[_tri.z]
                # print(ti_ver_3_tmp[2])
                _is_hit, _dir_t, _u, _v = _ray.intersect_triangle(self.ver_3_tmp)
                if _is_hit == 1:
                    _dir_t_max = max(_dir_t_max, _dir_t)
                    _dir_t_min = min(_dir_t_min, _dir_t)
            
            # 如果穿过了参与介质
            if _dir_t_max - _dir_t_min > 0.00001:
                _hit_p_max = _ray.origin + _dir_t_max * _ray.direction
                _hit_p_min = _ray.origin + _dir_t_min * _ray.direction
                # 最近-最远交点
                _inter_p_max = Interaction(_hit_p_max, _ray.direction)
                _inter_p_min = Interaction(_hit_p_min, _ray.direction)
                self.img[_i, _j] = self.ray_marching(_inter_p_min, _inter_p_max) * 0.01