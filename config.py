# config.py  ── Taichi 版（幾乎不變，加 ti.init）
import numpy as np
import taichi as ti

# --- 後端選擇：Colab/有 NVIDIA GPU → ti.gpu；純 CPU → ti.cpu ---
ti.init(arch=ti.gpu, default_fp=ti.f32)   # f32 在 GPU 上比 f64 快約 2x

# --- 物理與網格基本參數 ---
NX       = 1500
NY       = 375
MAX_STEPS = 35000
U_MAX    = 0.08
RE       = 500

# --- LBM D2Q9 常數（保留 NumPy 版，供 from_numpy 初始化用）---
CX_NP  = np.array([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=np.int32)
CY_NP  = np.array([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=np.int32)
W_NP   = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36], dtype=np.float32)
OPP_NP = np.array([ 0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# --- Taichi fields（GPU 上常駐）---
CX  = ti.field(ti.i32,  shape=9)
CY  = ti.field(ti.i32,  shape=9)
W   = ti.field(ti.f32,  shape=9)
OPP = ti.field(ti.i32,  shape=9)

def init_constants():
    CX.from_numpy(CX_NP)
    CY.from_numpy(CY_NP)
    W.from_numpy(W_NP)
    OPP.from_numpy(OPP_NP)

# --- 鬆弛參數 ---
L_char = 30.0
nu     = (U_MAX * L_char) / RE
tau    = 3.0 * nu + 0.5
OMEGA  = float(1.0 / tau)