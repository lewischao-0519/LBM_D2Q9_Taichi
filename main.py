# main.py - Kaggle Optimized Version (修正版)
import numpy as np
import os
import matplotlib
# 強制在無頭模式下運行，避免 Kaggle 報錯
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import config as cfg
from config import init_constants
from geometry import DomainManager
from engine import (f, f_new, rho_field, ux_field, uy_field, 
                    force_field_front, force_field_rear,
                    compute_force_dual_kernel, 
                    init_fields, lbm_step_kernel, swap_fields,
                    set_inlet_kernel, compute_macro_kernel)

def main():
    # 1. 初始化
    print("🔧 Initializing Taichi and constants...")
    init_constants()
    init_fields()
    
    domain = DomainManager(cfg.NX, cfg.NY)
    
    # 📝 修正：合理的機翼位置參數
    # 參數說明：(x_offset, y_offset, chord_length, thickness, angle_of_attack, label)
    # 
    # 前翼：放在 x=600 處，中心高度 y=600（域的一半），弦長 300，厚度 12%，攻角 -5°
    domain.add_naca_airfoil(
        x_offset=600,          # 前緣 x 位置
        y_offset=600,          # 中心線 y 位置  
        chord_length=300,      # 弦長
        t=0.12,                # NACA0012 (12% 厚度)
        angle_of_attack=-5.0,  # 攻角 -5°
        label=1                # 標記為前翼
    )
    
    # 後翼：放在 x=1500 處（距離前翼約 2 倍弦長），稍低一點 y=550，弦長 400，攻角 -2°
    domain.add_naca_airfoil(
        x_offset=1500,         # 前緣 x 位置
        y_offset=550,          # 中心線 y 位置
        chord_length=400,      # 弦長（比前翼大）
        t=0.12,                # NACA0012
        angle_of_attack=-2.0,  # 攻角 -2°
        label=2                # 標記為後翼
    )
    
    domain.upload()
    print(f"✅ Domain initialized: {cfg.NX}x{cfg.NY} grid")
    
    # --- 科學數據記錄器 ---
    steps_log = []
    front_lift, front_drag = [], []
    rear_lift, rear_drag = [], []
    
    # --- 影片錄製設定 ---
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(np.zeros((cfg.NY, cfg.NX)), cmap='jet', vmin=0, vmax=0.12, origin='lower')
    plt.colorbar(img, label='Velocity Magnitude')
    writer = animation.FFMpegWriter(fps=30)

    print("🚀 Starting simulation on Kaggle GPU...")

    # 開始錄製影片
    with writer.saving(fig, "simulation_result.mp4", dpi=100):
        for step in range(cfg.MAX_STEPS):
            # A. 設定邊界條件（修正：移除重複呼叫）
            # 前 2000 步速度從 0 緩慢升到 U_MAX
            current_u = cfg.U_MAX * min(1.0, step / 2000.0)
            
            # 前 1000 步加入擾動以觸發渦流
            perturb = 0.005 * np.sin(step * 0.1) if step < 1000 else 0.0
            
            set_inlet_kernel(current_u, perturb)
            
            # B. 核心演化
            lbm_step_kernel(domain.obstacle, cfg.OMEGA)
            swap_fields()
            
            # C. 每 20 步記錄一次受力數據
            if step % 20 == 0:
                compute_force_dual_kernel(domain.obstacle)
                
                # 讀取受力（注意：force_field 是 Vector[2]）
                f_front = force_field_front.to_numpy()  # shape: (2,) 或 array([fx, fy])
                f_rear = force_field_rear.to_numpy()
                
                steps_log.append(step)
                # 修正：直接存取陣列元素
                front_drag.append(f_front[0])  # Fx
                front_lift.append(f_front[1])  # Fy
                rear_drag.append(f_rear[0])
                rear_lift.append(f_rear[1])
                
            # D. 每 100 步擷取影片影格
            if step % 100 == 0:
                compute_macro_kernel(domain.obstacle)
                u_mag = np.sqrt(ux_field.to_numpy()**2 + uy_field.to_numpy()**2)
                img.set_array(u_mag)
                ax.set_title(f"Step: {step}")
                writer.grab_frame()
                
                if step % 1000 == 0:
                    print(f"Progress: {step}/{cfg.MAX_STEPS} | Front Lift: {front_lift[-1]:.4f}, Drag: {front_drag[-1]:.4f}")

    print("✅ Simulation finished! Preparing scientific data...")
    
    # 繪製受力對比圖
    try:
        if len(steps_log) > 0:
            # 創建一個 2x1 的佈局：上面看升力對比，下面看阻力對比
            fig_data, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # --- 上圖：升力 (Lift) 對比 ---
            ax1.plot(steps_log, front_lift, label='Front Wing', color='blue', linewidth=1.5)
            ax1.plot(steps_log, rear_lift, label='Rear Wing', color='cyan', linestyle='--', linewidth=1.5)
            ax1.set_ylabel('Lift Force (Lattice Units)')
            ax1.set_title('Tandem Wing Aerodynamic Performance Comparison')
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend(loc='upper right')

            # --- 下圖：阻力 (Drag) 對比 ---
            ax2.plot(steps_log, front_drag, label='Front Wing Drag', color='red', linewidth=1.5)
            ax2.plot(steps_log, rear_drag, label='Rear Wing Drag', color='orange', linestyle='--', linewidth=1.5)
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Drag Force (Lattice Units)')
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.savefig("tandem_force_comparison.png", dpi=250)
            print("✅ 雙翼對比圖已儲存至: tandem_force_comparison.png")
        else:
            print("⚠️ No data logged - check if step % 20 condition is met")

    except Exception as e:
        print(f"❌ Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()

    print("--- Process Complete ---")

if __name__ == "__main__":
    main()
