import numpy as np
import math

# 读取密度场数据
# d = np.load("D:/CFD/250frames/densityXl08_0234.vol.npy")
d = np.ones([64, 96, 64]) * 2.5




# 关于密度：当密度很小的时候，由于kappa会很大，整个微分方程会使 Id0 爆炸式增长。可能要借助新的理论，或者使用一些trick，例如使用mask

class ZeroVectorField:
    def __init__(self, de):
        shape = np.append(de.shape, 3)
        self.vector_filed = np.zeros(shape)


# 定义平行光类
class DistantLight:
    def __init__(self, intensity, chromaticity, direction):
        self.i = intensity
        self.chr = np.array(chromaticity)
        self.dir = np.array(direction)


# 定义初始亮度场类
class InitDiffusionField:
    def __init__(self, de):
        self.de = de
        shape = self.de.shape
        self.id0 = np.ones(shape)
        self.id1 = ZeroVectorField(self.de).vector_filed


# 定义参与介质类
class ParticipatingMedia:
    def __init__(self, dens, absorption_cross_section, scattering_cross_section, hg_g):
        self.den = dens
        self.sigma_a = absorption_cross_section
        self.sigma_s = scattering_cross_section
        self.sigma_t = absorption_cross_section + scattering_cross_section
        self.albedo = scattering_cross_section / (absorption_cross_section + scattering_cross_section)
        self.g = hg_g # ？

    def phase_function(self, cos_theta):
        return (1 / (4 * math.pi)) * ((1 - self.g ** 2) / (1 + self.g ** 2 - 2 * self.g * cos_theta))

    def get_mu(self, n):
        mu = 0.0
        # Riemann积分计算一阶矩
        for i in range(n + 1):
            cos_t = -1.0 + i * (2.0 / n)
            mu += cos_t * self.phase_function(cos_t)
        mu *= (3.0 / n)
        return mu

    def get_sigma_tr(self):
        mu_bar = self.get_mu(100)
        return self.sigma_s * (1 - mu_bar / 3.0) + self.sigma_a

    def init_light_field(self):
        diffusion_field = InitDiffusionField(self.den)
        return diffusion_field


# 定义梯度计算方法
def get_grid(matrix, h, option):
    grid_x = np.zeros(matrix.shape)
    grid_y = np.zeros(matrix.shape)
    grid_z = np.zeros(matrix.shape)
    grid = ZeroVectorField(matrix)
    result = grid.vector_filed
    # 精度低
    if option == 1:
        grid_x[0:-1, :, :] = (matrix[1:, :, :] - matrix[0:-1, :, :]) / h
        grid_x[-1, :, :] = grid_x[-2, :, :]
        grid_y[:, 0:-1, :] = (matrix[:, 1:, :] - matrix[:, 0:-1, :]) / h
        grid_y[:, -1, :] = grid_y[:, -2, :]
        grid_z[:, :, 0:-1] = (matrix[:, :, 1:] - matrix[:, :, 0:-1]) / h
        grid_x[:, :, -1] = grid_x[:, :, -2]
    # 精度高
    if option == 2:
        grid_x[1:-1, :, :] = (matrix[2:, :, :] - matrix[:-2, :, :]) / (2 * h)
        grid_x[0, :, :] = grid_x[1, :, :]
        grid_x[-1, :, :] = grid_x[-2, :, :]
        grid_y[:, 1:-1, :] = (matrix[:, 2:, :] - matrix[:, :-2, :]) / (2 * h)
        grid_y[:, 0, :] = grid_x[:, 1, :]
        grid_y[:, -1, :] = grid_x[:, -2, :]
        grid_z[:, :, 1:-1] = (matrix[:, :, 2:] - matrix[:, :, :-2]) / (2 * h)
        grid_z[:, :, 0] = grid_x[:, :, 1]
        grid_z[:, :, -1] = grid_x[:, :, -2]

    result[:, :, :, 0] = grid_x
    result[:, :, :, 1] = grid_y
    result[:, :, :, 2] = grid_z

    return result


# 定义循环节
def iterat(matrix, a, k, ss, h, option):
    # 这一版是错的，论文中少了一个括号
    if option == 1:
        matrix[1:-1, 1:-1, 1:-1] = (1 / (6 + 6 * k[1:-1, 1:-1, 1:-1] +
                                         a[1:-1, 1:-1, 1:-1] *
                                         h ** 2)) * \
                                   ((h ** 2) * ss[1:-1, 1:-1, 1:-1] +
                                    (k[1:-1, 1:-1, 1:-1] + k[2:, 1:-1, 1:-1]) *
                                    matrix[2:, 1:-1, 1:-1] +
                                    (k[1:-1, 1:-1, 1:-1] + k[:-2, 1:-1, 1:-1]) *
                                    matrix[:-2, 1:-1, 1:-1] +
                                    (k[1:-1, 1:-1, 1:-1] + k[1:-1, 2:, 1:-1]) *
                                    matrix[1:-1, 2:, 1:-1] +
                                    (k[1:-1, 1:-1, 1:-1] + k[1:-1, :-2, 1:-1]) *
                                    matrix[1:-1, :-2, 1:-1] +
                                    (k[1:-1, 1:-1, 1:-1] + k[1:-1, 1:-1, 2:]) *
                                    matrix[1:-1, 1:-1, 2:] +
                                    (k[1:-1, 1:-1, 1:-1] + k[1:-1, 1:-1, :-2]) *
                                    i_d_0[1:-1, 1:-1, :-2])
    # 这一版是也许正确的近似方程离散化
    if option == 2:
        matrix[1:-1, 1:-1, 1:-1] = ((1 / (6 + (h ** 2) * a[1:-1, 1:-1, 1:-1])) *
                                    ((h ** 2) * ss[1:-1, 1:-1, 1:-1] +
                                     (k[2:, 1:-1, 1:-1] * matrix[2:, 1:-1, 1:-1] +
                                      k[:-2, 1:-1, 1:-1] * matrix[:-2, 1:-1, 1:-1] +
                                      k[1:-1, 2:, 1:-1] * matrix[1:-1, 2:, 1:-1] +
                                      k[1:-1, :-2, 1:-1] * matrix[1:-1, :-2, 1:-1] +
                                      k[1:-1, 1:-1, 2:] * matrix[1:-1, 1:-1, 2:] +
                                      k[1:-1, 1:-1, :-2] * matrix[1:-1, 1:-1, :-2])))

    # 这一版是肯定正确的
    if option == 3:
        denominator = (h ** 2) * a[1:-1, 1:-1, 1:-1] + 6 * k[1:-1, 1:-1, 1:-1]
        sigma = ((k[2:, 1:-1, 1:-1] - k[:-2, 1:-1, 1:-1]) * (matrix[2:, 1:-1, 1:-1] - matrix[:-2, 1:-1, 1:-1]) +
                 (k[1:-1, 2:, 1:-1] - k[1:-1, :-2, 1:-1]) * (matrix[1:-1, 2:, 1:-1] - matrix[1:-1, :-2, 1:-1]) +
                 (k[1:-1, 1:-1, 2:] - k[1:-1, 1:-1, :-2]) * (matrix[1:-1, 1:-1, 2:] - matrix[1:-1, 1:-1, :-2]))
        sigma2 = (matrix[2:, 1:-1, 1:-1] + matrix[:-2, 1:-1, 1:-1] +
                  matrix[1:-1, 2:, 1:-1] + matrix[1:-1, :-2, 1:-1] +
                  matrix[1:-1, 1:-1, 2:] + matrix[1:-1, 1:-1, :-2])
        numerator = (h ** 2) * ss[1:-1, 1:-1, 1:-1] + sigma / 4 + k[1:-1, 1:-1, 1:-1] * sigma2
        matrix[1:-1, 1:-1, 1:-1] = numerator / denominator


# 定义方形介质边界条件
def rect_boundary(matrix, matrix1, k, h, fume):
    matrix[0, :, :] = ((2 * k[0, :, :] * matrix[1, :, :] - 2 *
                        h * ((fume.sigma_s / fume.get_sigma_tr()) * matrix1[0, :, :])) /
                       (h + 2 * k[0, :, :]))
    matrix[-1, :, :] = ((2 * k[-1, :, :] * matrix[-2, :, :] + 2 *
                         h * ((fume.sigma_s / fume.get_sigma_tr()) * matrix1[-1, :, :])) /
                        (h + 2 * k[-1, :, :]))
    matrix[:, 0, :] = (2 * k[:, 0, :] * matrix[:, 1, :] /
                       (h + 2 * k[:, 0, :]))
    matrix[:, -1, :] = (2 * k[:, -1, :] * matrix[:, -2, :] /
                        (h + 2 * k[:, -1, :]))
    matrix[:, :, 0] = (2 * k[:, :, 0] * matrix[:, :, 1] /
                       (h + 2 * k[:, :, 0]))
    matrix[:, :, -1] = (2 * k[:, :, -1] * matrix[:, :, -2] /
                        (h + 2 * k[:, :, -1]))


# 初始化场景参量
# 环境光：先实现一个简单的光照情况
l_env = DistantLight(1000.0, [0.7, 0.7, 0.7], [1.0, 0, 0])
# 参与介质
smoke = ParticipatingMedia(d, 0.25, 0.2, 0.4)
diffusionField = smoke.init_light_field()
# 网格步长
STEP_SIZE = 0.1

# 密度场
density = smoke.den
# Id0场
i_d_0 = diffusionField.id0
# Id1场
i_d_1 = diffusionField.id1
# kappa
kappa = 1.0 / (smoke.get_sigma_tr() * density)
# alpha
alpha = smoke.sigma_a * density
# S场
# Q_ri_0
# x 方向的transmittance
transmittance = np.zeros(density.shape)
for x in range(transmittance.shape[0]):
    transmittance[x, :, :] = np.exp(-smoke.sigma_t * STEP_SIZE * np.sum(density[0:x + 1, :, :], axis=0))
# I_ri
i_ri = l_env.i * transmittance
q_ri_0 = i_ri / (4 * math.pi)
# Q_ri_1 梯度
q_ri_1 = (smoke.get_mu(100) / (4 * math.pi)) * i_ri
grad_qri1 = np.zeros(q_ri_1.shape)
grad_qri1[0:-1, :, :] = q_ri_1[1:, :, :] - q_ri_1[0:-1, :, :]
s = smoke.sigma_s * density * q_ri_0 - (smoke.sigma_s / smoke.get_sigma_tr()) * grad_qri1

# TODO 使用V循环对算法提速
# 有限差分
rect_boundary(i_d_0, q_ri_1, kappa, STEP_SIZE, smoke)
for iteration in range(10 ** 6):
    i_d_0_last = i_d_0.copy()
    iterat(i_d_0, alpha, kappa, s, STEP_SIZE, 3)
    rect_boundary(i_d_0, q_ri_1, kappa, STEP_SIZE, smoke)
    error = np.max(abs(i_d_0 - i_d_0_last) / i_d_0)
    if error < 1.0e-6:
        print(f"id0 converged in {iteration + 1} steps")
        break
    if iteration % 50.0 == 0.0:
        print("it's the %d steps now, the error equals to %f" % (iteration, error))
        print(i_d_0.max())
        print(i_d_0.min())
else:
    print("Solutions failed to converged.")

i_d_1 = get_grid(i_d_0, STEP_SIZE, 2)
i_d_1[:, :, :, 0] = - kappa * i_d_1[:, :, :, 0] + smoke.sigma_s * density * q_ri_1
i_d_1[:, :, :, 1] = - kappa * i_d_1[:, :, :, 1]
i_d_1[:, :, :, 2] = - kappa * i_d_1[:, :, :, 2]

np.save('id1', i_d_1)
np.save('id0', i_d_0)

# 最后的结果：i_d_0 和 i_d_1
# TODO 使用 Ray marching 计算随后的像素值
