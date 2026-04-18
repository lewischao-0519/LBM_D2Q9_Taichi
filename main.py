# main.py - Kaggle Optimized Version
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
from engine import (f, f_new, rho_field, ux_field, uy_field, force_field,
                    init_fields, lbm_step_kernel, swap_fields,
                    set_inlet_kernel, compute_macro_kernel, compute_force_kernel)

def main():
    # 1. 初始化
    init_constants()
    init_fields()
    
    domain = DomainManager(cfg.NX, cfg.NY)
    
    # 📝 修正：給予明確的數值而非佔位符
    # 參數：(x位置, y位置, 弦長, 厚度t, 攻角)
    domain.add_naca_airfoil(200, 400, 300, 0.12, -5.0,label=1)
    domain.add_naca_airfoil(1300, 360, 400, 0.12, -2.0,label=2)
    domain.upload()
    
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
            # A. 設定邊界擾動 (讓渦流更快產生)
            # 在 for step in range(cfg.MAX_STEPS) 循環內
            # 前 2000 步速度從 0 緩慢升到 U_MAX
            current_u = cfg.U_MAX * min(1.0, step / 2000.0)
            set_inlet_kernel(current_u, 0.0)
            perturb = 0.005 * np.sin(step * 0.1) if step < 1000 else 0.0
            set_inlet_kernel(cfg.U_MAX, perturb)
            
            # B. 核心演化
            lbm_step_kernel(domain.obstacle, cfg.OMEGA)
            swap_fields()
            
            # C. 每 20 步記錄一次受力數據
            if step % 20 == 0:
                compute_force_dual_kernel(domain.obstacle)
                f_front = force_field_front.to_numpy()
                f_rear = force_field_rear.to_numpy()
                
                steps_log.append(step)
                front_drag.append(f_front[0]); front_lift.append(f_front[1])
                rear_drag.append(f_rear[0]); rear_lift.append(f_rear[1])
                
            # D. 每 100 步擷取影片影格
            if step % 100 == 0:
                compute_macro_kernel(domain.obstacle)
                u_mag = np.sqrt(ux_field.to_numpy()**2 + uy_field.to_numpy()**2)
                img.set_array(u_mag)
                ax.set_title(f"Step: {step}")
                writer.grab_frame()
                
                if step % 1000 == 0:
                    print(f"Current Progress: {step}/{cfg.MAX_STEPS}")

    print("✅ Simulation finished! Preparing scientific data...")
    
    try:
        if len(steps_log) > 0:
        # 創建一個 2x1 的佈局：上面看升力對比，下面看阻力對比
            fig_data, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # --- 上圖：升力 (Lift) 對比 ---
            ax1.plot(steps_log, front_lift, label='Front Wing (Target)', color='blue', linewidth=1.5)
            ax1.plot(steps_log, rear_lift, label='Rear Wing (Tandem)', color='cyan', linestyle='--', linewidth=1.5)
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
            plt.savefig("/kaggle/working/tandem_force_comparison.png", dpi=250)
            print("✅ 雙翼對比圖已儲存至: tandem_force_comparison.png")

    except Exception as e:
        print(f"❌ Failed to generate plot: {e}")

    print("--- Process Complete ---")

# ✨ 關鍵修正：必須呼叫主程式
if __name__ == "__main__":
    main()