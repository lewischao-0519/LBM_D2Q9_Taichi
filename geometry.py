# geometry.py  ── Taichi 版（obstacle 改為 ti.field）
import numpy as np
import taichi as ti
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

    def clear_domain(self):
        self._mask_np.fill(0)

    def upload(self):
        """把 NumPy mask 搬到 GPU（只需呼叫一次）"""
        self.obstacle.from_numpy(self._mask_np)

    # 保留舊介面，方便 main.py 相容
    def get_obstacle_mask(self):
        return self._mask_np
