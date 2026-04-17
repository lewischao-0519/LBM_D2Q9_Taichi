# engine.py  ── Taichi 版（核心遷移）
#
# 對應原版 Numba @njit(parallel=True) lbm_step()
# 改為三個 @ti.kernel：lbm_step_kernel / set_inlet_kernel / get_macro_kernel
#
import taichi as ti
import numpy as np
import config as cfg

# ── 全域分佈函數 field（GPU 記憶體常駐）──
f     = ti.field(ti.f32, shape=(9, cfg.NY, cfg.NX))
f_new = ti.field(ti.f32, shape=(9, cfg.NY, cfg.NX))

# ── 可視化用宏觀量（供 main.py to_numpy() 讀出）──
rho_field = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
ux_field  = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
uy_field  = ti.field(ti.f32, shape=(cfg.NY, cfg.NX))
# engine.py 

# 建立一個零維向量場來儲存總受力 (只有一個元素，存放 [Fx, Fy])
force_field = ti.Vector.field(2, dtype=ti.f32, shape=())

@ti.kernel
def compute_force_kernel(obstacle: ti.template()):
    force_field[None] = [0.0, 0.0]
    for y, x in ti.ndrange(cfg.NY, cfg.NX):
        if obstacle[y, x] == 1:
            for i in ti.static(range(9)):
                nb_x = (x + cfg.CX[i]) % cfg.NX
                nb_y = (y + cfg.CY[i]) % cfg.NY
                if obstacle[nb_y, nb_x] == 0:
                    # 動量交換公式
                    f_in = f[i, nb_y, nb_x]
                    f_out = f[cfg.OPP[i], y, x]
                    force_field[None] += ti.Vector([cfg.CX[i], cfg.CY[i]]) * (f_in + f_out)


@ti.kernel
def lbm_step_kernel(obstacle: ti.template(), omega: float):
    """
    等價於原版 lbm_step()：
      - Pull-scheme 串流
      - 碰撞（BGK / Loop Fusion）
      - Bounce-back 反彈
    Taichi 對 for y, x in ... 自動並行，不需要 prange。
    """
    for y, x in rho_field:          # 自動並行所有 (y, x)

        # ── Bounce-back（障礙物）──
        if obstacle[y, x] == 1:
            for i in ti.static(range(9)):
                f_new[i, y, x] = f[cfg.OPP[i], y, x]
            # 注意：Taichi kernel 裡不能 continue，用 else 繞過
        else:
            # ── Pull-scheme 串流 ──
            f_local = ti.Vector([0.0] * 9)
            for i in ti.static(range(9)):
                prev_x = (x - cfg.CX[i] + cfg.NX) % cfg.NX
                prev_y = (y - cfg.CY[i] + cfg.NY) % cfg.NY
                f_local[i] = f[i, prev_y, prev_x]

            # ── 宏觀量 ──
            r  = 0.0
            vx = 0.0
            vy = 0.0
            for i in ti.static(range(9)):
                r  += f_local[i]
                vx += f_local[i] * cfg.CX[i]
                vy += f_local[i] * cfg.CY[i]

            r  = max(r, 1e-6)
            vx /= r
            vy /= r

            # ── BGK 碰撞（Loop Fusion）──
            u_sq = vx * vx + vy * vy
            for i in ti.static(range(9)):
                cu  = cfg.CX[i] * vx + cfg.CY[i] * vy
                feq = r * cfg.W[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq)
                f_new[i, y, x] = f_local[i] * (1.0 - omega) + feq * omega


@ti.kernel
def swap_fields():
    """交換 f 與 f_new（等價於原版 f, f_new = f_new, f）"""
    for i, y, x in f:
        f[i, y, x], f_new[i, y, x] = f_new[i, y, x], f[i, y, x]


@ti.kernel
def set_inlet_kernel(u_max: float, perturbation: float):
    """
    等價於原版 set_inlet_boundary()。
    直接在 GPU 上設定 x=0 的入口邊界，省去 CPU↔GPU 搬資料。
    """
    for y in range(cfg.NY):
        rho_in = 1.0
        ux_in  = u_max
        uy_in  = perturbation
        u_sq   = ux_in * ux_in + uy_in * uy_in
        for i in ti.static(range(9)):
            cu  = cfg.CX[i] * ux_in + cfg.CY[i] * uy_in
            feq = rho_in * cfg.W[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*u_sq)
            f[i, y, 0] = feq


@ti.kernel
def compute_macro_kernel(obstacle: ti.template()):
    """
    計算宏觀量並存入 rho_field / ux_field / uy_field，
    障礙物內部速度設為 0。
    """
    for y, x in rho_field:
        if obstacle[y, x] == 1:
            ux_field[y, x]  = 0.0
            uy_field[y, x]  = 0.0
            rho_field[y, x] = 1.0
        else:
            r  = 0.0
            vx = 0.0
            vy = 0.0
            for i in ti.static(range(9)):
                r  += f[i, y, x]
                vx += f[i, y, x] * cfg.CX[i]
                vy += f[i, y, x] * cfg.CY[i]
            r = max(r, 1e-6)
            rho_field[y, x] = r
            ux_field[y, x]  = vx / r
            uy_field[y, x]  = vy / r


def init_fields():
    """初始化 f / f_new 為均勻流場（等價於原版 init_simulation 的 f 部分）"""
    for i in range(9):
        val = float(cfg.W_NP[i])
        # 用 numpy 批次填入
        arr = np.full((cfg.NY, cfg.NX), val, dtype=np.float32)
        f.from_numpy(f.to_numpy())          # 確保 shape 一致
    # 更直接的寫法：
    f_np = np.zeros((9, cfg.NY, cfg.NX), dtype=np.float32)
    for i in range(9):
        f_np[i] = cfg.W_NP[i]
    f.from_numpy(f_np)
    f_new.from_numpy(f_np)