import taichi as ti
import numpy as np
# 乱七八糟、大杂烩，不知道怎么分类的都放这了

infinite = 1e5
infinitesimal = 1e-5

# 读取off文件
def readOFF(file_path):
    _ver = None
    _tri = None
    _count = 0
    with open(file_path, 'r') as _f:
        for _line in _f.readlines():
            _line = _line.strip('\n').split(' ')
            if len(_line) == 3 and _count != 1:
                if _ver is None:
                    _ver = np.array([_line])
                else:
                    _ver = np.append(_ver, np.array([_line]), axis = 0)
                # _ver.append(_line)
            elif len(_line) == 4 and _line[0] == '3':
                if _tri is None:
                    _tri = np.array([_line[1:4]])
                else:
                    _tri = np.append(_tri, np.array([_line[1:4]]), axis = 0)
            _count += 1
    return _ver.astype(np.float32), _tri.astype(np.int32)

# debug打印向量
@ti.func
def printVector(vec):
    for i in ti.ndrange(vec.shape[0]):
        print(vec[i])

# 四舍五入-标量
@ti.func
def round(val):
    # 向上取整
    _round_val = ti.ceil(val)
    if (_round_val - val) > 0.5:
        _round_val -= 1.0
    return _round_val
# 四舍五入-向量
@ti.func
def round_vector(val):
    _val = val
    # 四舍五入，获取p点所在体素位置，整数
    for i in ti.static(range(_val.n)):
        _val[i] = round(_val[i])
    return _val

