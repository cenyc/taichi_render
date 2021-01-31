import taichi as ti

@ti.data_oriented
class Ray:
    # org-光线起点，dir-光线方向
    def __init__(self, org, dir):
        self.origin = org
        self.direction = dir

    # 光线与三角形求交算法
    # ver 三角形顶点位置
    # 算法参考：https://www.cnblogs.com/graphics/archive/2010/08/09/1795348.html
    @ti.func
    def intersect_triangle(self, ver):
        _is_hit = 1; _dir_t = 0.0; _u = 0.0; _v = 0.0
        # E1
        _e1 = ver[1] - ver[0]
        # E2
        _e2 = ver[2] - ver[0]
        # P
        _p = self.direction.cross(_e2)
        # determinant
        _det = _e1.dot(_p)
        _t = ti.Vector([0.0, 0.0, 0.0])
        if _det > 0:
            _t = self.origin - ver[0]
        else:
            _t = ver[0] - self.origin
            _det = -_det

        if _det < 0.0001:
            _is_hit = 0
        else:
            _u = _t.dot(_p)

            if _u < 0.0 or _u > _det:
                _is_hit = 0
            else:
                _q = _t.cross(_e1)
                _v = self.direction.dot(_q)
                if _v < 0.0 or _u+_v >_det:
                    _is_hit = 0
                else:
                    _dir_t = _e2.dot(_q)
                    _fInvDet = 1.0/_det
                    _dir_t *= _fInvDet
                    _u *= _fInvDet
                    _v *= _fInvDet

        return _is_hit, _dir_t, _u, _v