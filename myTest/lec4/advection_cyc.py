import taichi as ti

ti.init(arch=ti.cpu)

n = 512
pause = False
rk = 2
use_mc = True
mc_clipping = True

x = ti.field(ti.f32, shape=(n, n))
new_x = ti.field(ti.f32, shape=(n, n))
new_x_aux = ti.field(ti.f32, shape=(n, n))
dt = 0.05
stagger = ti.Vector([0.5, 0.5])
dx = 1.0/n
inv_dx = 1/dx

@ti.func
def Vector2(x, y):
    return ti.Vector([x, y])

@ti.func
def inside(p, c, r):
    return (p - c).norm_sqr() <= r * r

@ti.func
def velocity(p):
    return ti.Vector([p[1] - 0.5, 0.5 - p[0]])

@ti.func
def inside_taichi(p):
    p = Vector2(0.5, 0.5) + (p - Vector2(0.5, 0.5)) * 1.2
    ret = -1
    if not inside(p, Vector2(0.50, 0.50), 0.55):
        if ret == -1:
            ret = 0
    if not inside(p, Vector2(0.50, 0.50), 0.50):
        if ret == -1:
            ret = 1
    if inside(p, Vector2(0.50, 0.25), 0.09):
        if ret == -1:
            ret = 1
    if inside(p, Vector2(0.50, 0.75), 0.09):
        if ret == -1:
            ret = 0
    if inside(p, Vector2(0.50, 0.25), 0.25):
        if ret == -1:
            ret = 0
    if inside(p, Vector2(0.50, 0.75), 0.25):
        if ret == -1:
            ret = 1
    if p[0] < 0.5:
        if ret == -1:
            ret = 1
    else:
        if ret == -1:
            ret = 0
    return ret

@ti.func
def inside_half(p):
    ret = 0
    if p[0] < 0.5:
        ret = 1
    return ret

@ti.kernel
def paint():
    for i, j in ti.ndrange(n * 4, n * 4):
        ret = 1 - inside_taichi(Vector2(i / n / 4, j / n / 4))
        # ret = inside_half(Vector2(i / n / 4, j / n / 4))
        x[i // 4, j // 4] += ret / 16

@ti.func
def backtrace(I, dt):
    # 这里认为每个格子的速度在该格子的中点，即在该格子中的所有粒子，都使用这个速度进行backtrace计算
    p = (I+stagger)*dx

    if rk == 1:
        p -= dt*velocity(p)
    elif rk == 2:
        p_mid = p - dt*0.5*velocity(p)
        p -= dt*velocity(p_mid)
    return p

@ti.func
def vec(x, y):
    return ti.Vector([x, y])
@ti.func

def clamp(p):
    for d in ti.static(range(p.n)):
        p[d] = min(1 - 1e-4 - dx + stagger[d] * dx, max(p[d], stagger[d]*dx))
    return p

@ti.func
def sample_bilinear(x, p):
    p = clamp(p)

    # 求p在哪个格子
    p_grid = p*inv_dx - stagger
    # 向下取整
    I = ti.cast(ti.floor(p_grid), ti.int32)
    f = p_grid - I
    g = 1 - f

    return x[I] * (g[0]*g[1]) + x[I+vec(1, 0)] * (g[1]*f[0]) + \
        x[I+vec(0, 1)] * (g[0]*f[1]) + x[I+vec(1,1)] * (f[0]*f[1])

@ti.func
def semi_lagrangian(x, new_x, dt):
    for I in ti.grouped(x):
        new_x[I] = sample_bilinear(x, backtrace(I, dt))


@ti.func
def sample_min(x, p):
    p = clamp(p)
    p_grid = p*inv_dx - stagger
    I = ti.cast(ti.floor(p_grid), ti.f32)
    return min(x[I],  x[I + vec(1, 0)], x[I + vec(0, 1)], x[I + vec(1, 1)])

@ti.func
def sample_max(x, p):
    p = clamp(p)
    p_grid = p * inv_dx - stagger
    I = ti.cast(ti.floor(p_grid), ti.f32)
    return max(x[I], x[I + vec(1, 0)], x[I + vec(0, 1)], x[I + vec(1, 1)])

@ti.func
def maccormick(x, dt):
    # 往前推dt时间
    semi_lagrangian(x, new_x, dt)
    # 往后推dt时间
    semi_lagrangian(new_x, new_x_aux, -dt)
    for I in ti.grouped(x):
        # 这个修正项有时候会弄巧成拙，产生了类似吉布斯现象的伪影
        new_x[I] += 0.5*(x[I] - new_x_aux[I])
        if mc_clipping:
            source_pos = backtrace(I, dt)
            min_val = sample_min(x, source_pos)
            max_val = sample_max(x, source_pos)

            if new_x[I] < min_val or new_x[I] > max_val:
                new_x[I] = sample_bilinear(x, source_pos)

@ti.kernel
def advect():
    if use_mc:
        maccormick(x, dt)
    else:
        semi_lagrangian(x, new_x, dt)
    for I in ti.grouped(x):
        x[I] = new_x[I]
paint()
gui = ti.GUI('Advection schemes', (512, 512))
while True:
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: exit(0)
        if gui.event.key == ti.GUI.SPACE:
            pause = not pause

    if not pause:
        for i in range(1):
            advect()

    gui.set_image(x.to_numpy())
    gui.show()