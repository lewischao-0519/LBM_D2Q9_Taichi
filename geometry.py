# geometry.py  ── Taichi 版（obstacle 改為 ti.field）
import numpy as np
import taichi as ti
import config as cfg
from config import NX, NY

class DomainManager:
    """
    管理計算域內的障礙物。
    內部仍用 NumPy 組裝，最後一次性搬進 GPU。
    未來電漿擴展（電極、磁場邊界）同樣在這裡加欄位。
    """
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self._mask_np = np.zeros((ny, nx), dtype=np.int32)  # 用 int32，Taichi 較友好

        # ── Taichi field（GPU 上常駐）──
        self.obstacle = ti.field(ti.i32, shape=(ny, nx))

        # 電漿擴展預留：
        # self.electric_potential = ti.field(ti.f32, shape=(ny, nx))

    def add_cylinder(self, cx, cy, r):
        Y, X = np.ogrid[:self.ny, :self.nx]
        self._mask_np |= ((X - cx)**2 + (Y - cy)**2 <= r**2).astype(np.int32)

    def add_rectangle(self, x_start, x_end, y_start, y_end):
        self._mask_np[y_start:y_end, x_start:x_end] = 1
    
    def add_naca_airfoil(self, x_offset, y_offset, chord_length, t, angle_of_attack, label):
        # --- 1. 定義幾何判斷函數 (必須在迴圈之前) ---
        def is_inside_airfoil(px, py):
            # 將座標平移並旋轉回機翼本地座標系
            dx = px - x_offset
            dy = py - y_offset
            cos_a = np.cos(np.radians(-angle_of_attack))
            sin_a = np.sin(np.radians(-angle_of_attack))
            
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            
            # NACA 4-digit 厚度分布公式
            if 0 <= rx <= chord_length:
                x_norm = rx / chord_length
                yt = 5 * t * chord_length * (
                    0.2969 * np.sqrt(x_norm) - 
                    0.1260 * x_norm - 
                    0.3516 * x_norm**2 + 
                    0.2843 * x_norm**3 - 
                    0.1015 * x_norm**4
                )
                return abs(ry) <= yt
            return False

        # --- 2. 執行遍歷與標記 ---
        # 建議只在機翼可能的範圍內遍歷 (Bounding Box)，加速啟動
        x_start = max(0, int(x_offset - chord_length * 0.2))
        x_end = min(cfg.NX, int(x_offset + chord_length * 1.2))
        y_start = max(0, int(y_offset - chord_length * 0.5))
        y_end = min(cfg.NY, int(y_offset + chord_length * 0.5))

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                if is_inside_airfoil(i, j):
                    self.obstacle[i, j] = label # 使用傳入的 label (1 或 2)

    def clear_domain(self):
        self._mask_np.fill(0)

    def upload(self):
        """把 NumPy mask 搬到 GPU（只需呼叫一次）"""
        self.obstacle.from_numpy(self._mask_np)

    # 保留舊介面，方便 main.py 相容
    def get_obstacle_mask(self):
        return self._mask_np