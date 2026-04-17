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
    # 這裡是你的飛機機翼或圓柱設定
    domain.add_naca_airfoil(self, x_offset, y_offset, chord_length, t, angle_of_attack)
    domain.upload()
    
    # --- 科學數據記錄器 ---
    steps_log = []
    lift_log = []
    drag_log = []
    
    # --- 影片錄製設定 ---
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(np.zeros((cfg.NY, cfg.NX)), cmap='jet', vmin=0, vmax=0.12, origin='lower')
    plt.colorbar(img, label='Velocity Magnitude')
    writer = animation.FFMpegWriter(fps=30)

    print("Starting simulation on Kaggle GPU...")

    # 開始錄製影片
    with writer.saving(fig, "simulation_result.mp4", dpi=100):
        for step in range(cfg.MAX_STEPS):
            # A. 設定邊界擾動
            perturb = 0.005 * np.sin(step * 0.1) if step < 1000 else 0.0
            set_inlet_kernel(cfg.U_MAX, perturb)
            
            # B. 核心演化
            lbm_step_kernel(domain.obstacle, cfg.OMEGA)
            swap_fields()
            
            # C. 每 20 步記錄一次受力數據 (Fx, Fy)
            if step % 20 == 0:
                compute_force_kernel(domain.obstacle)
                force = force_field.to_numpy() # 從 GPU 拉回數值
                steps_log.append(step)
                drag_log.append(force[0])
                lift_log.append(force[1])
            
            # D. 每 100 步擷取影片影格
            if step % 100 == 0:
                compute_macro_kernel(domain.obstacle)
                u_mag = np.sqrt(ux_field.to_numpy()**2 + uy_field.to_numpy()**2)
                img.set_array(u_mag)
                ax.set_title(f"Step: {step}")
                writer.grab_frame()
                
                if step % 1000 == 0:
                    print(f"Current Progress: {step}/{cfg.MAX_STEPS}")

# ... 前方的模擬與錄影代碼 ...

print("Simulation finished! Preparing scientific data...")

# 強制開啟一個全新的 Figure，避免受影片錄製影響
try:
    if len(steps_log) > 0:
        plt.clf() # 清除所有目前的繪圖內容
        fig_data, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 繪製升力 Fy
        ax1.plot(steps_log, lift_log, label='Lift (Fy)', color='blue', linewidth=1.5)
        ax1.set_ylabel('Force (Lattice Units)')
        ax1.set_title('Aerodynamic Force Analysis (Kármán Vortex Street)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # 繪製阻力 Fx
        ax2.plot(steps_log, drag_log, label='Drag (Fx)', color='red', linewidth=1.5)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Force (Lattice Units)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        
        # 儲存圖片
        save_path = "/kaggle/working/force_analysis_plot.png"
        plt.savefig(save_path, dpi=200)
        print(f"✅ Scientific plot saved successfully to: {save_path}")
        
        # (選配) 將數據存成 CSV，這才是真正的「科學數據」
        import pandas as pd
        df = pd.DataFrame({
            'step': steps_log,
            'drag': drag_log,
            'lift': lift_log
        })
        df.to_csv("/kaggle/working/force_data.csv", index=False)
        print("✅ Data exported to force_data.csv")
        
    else:
        print("❌ Error: No force data was recorded. Check your step % 20 condition.")

except Exception as e:
    print(f"❌ Failed to generate plot: {e}")

    print("--- Process Complete ---")