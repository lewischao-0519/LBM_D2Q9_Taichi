# geometry.py  ── Taichi 版（修正版）
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
        self._mask_np = np.zeros((ny, nx), dtype=np.int32)  # [y, x] 順序

        # ── Taichi field（GPU 上常駐）──
        self.obstacle = ti.field(ti.i32, shape=(ny, nx))  # [y, x] 順序

    def add_cylinder(self, cx, cy, r):
        """添加圓柱障礙物"""
        Y, X = np.ogrid[:self.ny, :self.nx]
        self._mask_np |= ((X - cx)**2 + (Y - cy)**2 <= r**2).astype(np.int32)

    def add_rectangle(self, x_start, x_end, y_start, y_end):
        """添加矩形障礙物"""
        self._mask_np[y_start:y_end, x_start:x_end] = 1
    
    def add_naca_airfoil(self, x_offset, y_offset, chord_length, t, angle_of_attack, label):
        """
        添加 NACA 4-digit 翼型
        
        參數:
            x_offset: 翼型前緣的 x 座標
            y_offset: 翼型中心線的 y 座標
            chord_length: 弦長（像素）
            t: 厚度參數（例如 0.12 代表 NACA0012）
            angle_of_attack: 攻角（度）
            label: 障礙物標籤（1=前翼, 2=後翼）
        """
        # --- 1. 定義幾何判斷函數 ---
        def is_inside_airfoil(px, py):
            """判斷點 (px, py) 是否在翼型內部"""
            # 將座標平移並旋轉回機翼本地座標系
            dx = px - x_offset
            dy = py - y_offset
            cos_a = np.cos(np.radians(-angle_of_attack))
            sin_a = np.sin(np.radians(-angle_of_attack))
            
            # 旋轉到機翼座標系
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

        # --- 2. 執行遍歷與標記（只在 Bounding Box 內）---
        # 估算旋轉後的包圍盒
        padding = chord_length * 0.6
        x_start = max(0, int(x_offset - padding))
        x_end = min(self.nx, int(x_offset + chord_length + padding))
        y_start = max(0, int(y_offset - padding))
        y_end = min(self.ny, int(y_offset + padding))

        # 遍歷像素並標記（注意：NumPy array 是 [y, x] 順序）
        for j in range(y_start, y_end):  # j = y 座標
            for i in range(x_start, x_end):  # i = x 座標
                if is_inside_airfoil(i, j):  # 傳入 (x, y)
                    self._mask_np[j, i] = label  # 存儲時用 [y, x]

    def clear_domain(self):
        """清空整個計算域"""
        self._mask_np.fill(0)

    def upload(self):
        """把 NumPy mask 搬到 GPU（只需呼叫一次）"""
        self.obstacle.from_numpy(self._mask_np)

    def get_obstacle_mask(self):
        """返回 NumPy 版本的障礙物遮罩（用於檢查或視覺化）"""
        return self._mask_np
