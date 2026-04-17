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
    domain.add_cylinder(cx=150, cy=75, r=15)
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

    print("Simulation finished! Generating plots...")

    # --- 繪製科學數據圖 ---
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(steps_log, lift_log, label='Lift (Fy)', color='blue')
    plt.ylabel('Force (Lattice Units)')
    plt.title('Aerodynamic Force Analysis')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(steps_log, drag_log, label='Drag (Fx)', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Force (Lattice Units)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("force_analysis_plot.png") # 儲存在 Kaggle Working 目錄
    print("Files saved: simulation_result.mp4, force_analysis_plot.png")

if __name__ == "__main__":
    main()