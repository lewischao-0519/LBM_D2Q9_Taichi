# main.py - 雙 T4 GPU 優化版本
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 無頭模式
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from datetime import datetime, timedelta

import config as cfg
from config import init_constants
from geometry import DomainManager
from engine import (f, f_new, rho_field, ux_field, uy_field, 
                    force_field_front, force_field_rear,
                    compute_force_dual_kernel, 
                    init_fields, lbm_step_kernel, swap_fields,
                    set_inlet_kernel, compute_macro_kernel)

def format_time(seconds):
    """格式化時間顯示"""
    return str(timedelta(seconds=int(seconds)))

def main():
    print("=" * 70)
    print("🚀 雙 T4 GPU LBM 雙翼模擬")
    print("=" * 70)
    
    # ========================================
    # 1. 初始化
    # ========================================
    start_time = time.time()
    
    print("\n[1/6] 初始化 Taichi GPU 常數...")
    init_constants()
    init_fields()
    
    print("[2/6] 創建計算域與機翼幾何...")
    domain = DomainManager(cfg.NX, cfg.NY)
    
    # 使用配置文件中的參數
    domain.add_naca_airfoil(
        x_offset=cfg.FRONT_WING_X,
        y_offset=cfg.FRONT_WING_Y,
        chord_length=cfg.FRONT_CHORD,
        t=cfg.FRONT_THICKNESS,
        angle_of_attack=cfg.FRONT_AOA,
        label=1  # 前翼
    )
    
    domain.add_naca_airfoil(
        x_offset=cfg.REAR_WING_X,
        y_offset=cfg.REAR_WING_Y,
        chord_length=cfg.REAR_CHORD,
        t=cfg.REAR_THICKNESS,
        angle_of_attack=cfg.REAR_AOA,
        label=2  # 後翼
    )
    
    domain.upload()
    init_time = time.time() - start_time
    print(f"✓ 初始化完成 (耗時: {format_time(init_time)})")
    
    # ========================================
    # 2. 準備數據記錄器
    # ========================================
    print("\n[3/6] 準備數據記錄器...")
    steps_log = []
    front_lift, front_drag = [], []
    rear_lift, rear_drag = [], []
    
    # 性能監控
    step_times = []
    mlups_history = []
    
    # ========================================
    # 3. 準備視覺化
    # ========================================
    print("[4/6] 準備影片錄製...")
    fig, ax = plt.subplots(figsize=(14, 4))
    img = ax.imshow(
        np.zeros((cfg.NY, cfg.NX)), 
        cmap='turbo',  # 更鮮豔的色彩映射
        vmin=0, 
        vmax=0.15,  # 稍微調高以看清高速區
        origin='lower',
        interpolation='bilinear'
    )
    cbar = plt.colorbar(img, label='Velocity Magnitude', ax=ax)
    ax.set_xlabel('X (lattice units)')
    ax.set_ylabel('Y (lattice units)')
    
    writer = animation.FFMpegWriter(fps=30, bitrate=5000)
    
    # ========================================
    # 4. 主模擬循環
    # ========================================
    print("\n[5/6] 開始主模擬循環...")
    print(f"目標步數: {cfg.MAX_STEPS:,}")
    print("=" * 70)
    
    sim_start_time = time.time()
    
    with writer.saving(fig, "tandem_wing_simulation.mp4", dpi=150):
        for step in range(cfg.MAX_STEPS):
            step_start = time.time()
            
            # A. 邊界條件（速度漸增 + 初期擾動）
            ramp_factor = min(1.0, step / 2000.0)  # 前 2000 步漸增
            current_u = cfg.U_MAX * ramp_factor
            
            # 前 1000 步加入週期擾動以觸發渦流
            if step < 1000:
                perturb = 0.01 * np.sin(step * 0.15)
            else:
                perturb = 0.0
            
            set_inlet_kernel(current_u, perturb)
            
            # B. LBM 演化
            lbm_step_kernel(domain.obstacle, cfg.OMEGA)
            swap_fields()
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # 計算 MLUPS（每秒百萬格點更新數）
            mlups = (cfg.NX * cfg.NY / 1e6) / step_time
            mlups_history.append(mlups)
            
            # C. 記錄受力數據
            if step % cfg.FORCE_LOG_INTERVAL == 0:
                compute_force_dual_kernel(domain.obstacle)
                
                f_front = force_field_front.to_numpy()
                f_rear = force_field_rear.to_numpy()
                
                steps_log.append(step)
                front_drag.append(f_front[0])
                front_lift.append(f_front[1])
                rear_drag.append(f_rear[0])
                rear_lift.append(f_rear[1])
            
            # D. 保存影片幀
            if step % cfg.SAVE_INTERVAL == 0:
                compute_macro_kernel(domain.obstacle)
                u_mag = np.sqrt(ux_field.to_numpy()**2 + uy_field.to_numpy()**2)
                
                img.set_array(u_mag)
                ax.set_title(
                    f"Step: {step:,} / {cfg.MAX_STEPS:,} | "
                    f"Re: {cfg.RE:,} | "
                    f"MLUPS: {mlups:.1f}",
                    fontsize=10
                )
                writer.grab_frame()
            
            # E. 進度報告
            if step % cfg.PRINT_INTERVAL == 0 and step > 0:
                elapsed = time.time() - sim_start_time
                avg_mlups = np.mean(mlups_history[-100:])  # 最近 100 步平均
                progress = step / cfg.MAX_STEPS
                eta = (elapsed / progress) * (1 - progress)
                
                print(f"Step {step:7,} / {cfg.MAX_STEPS:,} ({progress*100:5.2f}%) | "
                      f"MLUPS: {avg_mlups:6.1f} | "
                      f"Elapsed: {format_time(elapsed)} | "
                      f"ETA: {format_time(eta)}")
                print(f"  Front - Lift: {front_lift[-1]:8.4f}, Drag: {front_drag[-1]:8.4f}")
                print(f"  Rear  - Lift: {rear_lift[-1]:8.4f}, Drag: {rear_drag[-1]:8.4f}")
                print("-" * 70)
    
    sim_time = time.time() - sim_start_time
    avg_mlups = np.mean(mlups_history)
    
    print("\n" + "=" * 70)
    print("✅ 模擬完成！")
    print(f"總時長: {format_time(sim_time)}")
    print(f"平均性能: {avg_mlups:.1f} MLUPS")
    print(f"峰值性能: {max(mlups_history):.1f} MLUPS")
    print("=" * 70)
    
    # ========================================
    # 5. 後處理與繪圖
    # ========================================
    print("\n[6/6] 生成科學數據圖表...")
    
    try:
        if len(steps_log) > 0:
            # 創建 2×2 子圖佈局
            fig_data, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # --- 子圖 1: 升力對比 ---
            ax1.plot(steps_log, front_lift, label='Front Wing', 
                    color='#2E86AB', linewidth=1.5)
            ax1.plot(steps_log, rear_lift, label='Rear Wing', 
                    color='#A23B72', linestyle='--', linewidth=1.5)
            ax1.set_ylabel('Lift Force (Lattice Units)', fontsize=11)
            ax1.set_title('Lift Force Comparison', fontsize=12, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.4)
            ax1.legend(loc='upper right', fontsize=10)
            
            # --- 子圖 2: 阻力對比 ---
            ax2.plot(steps_log, front_drag, label='Front Wing', 
                    color='#F18F01', linewidth=1.5)
            ax2.plot(steps_log, rear_drag, label='Rear Wing', 
                    color='#C73E1D', linestyle='--', linewidth=1.5)
            ax2.set_ylabel('Drag Force (Lattice Units)', fontsize=11)
            ax2.set_title('Drag Force Comparison', fontsize=12, fontweight='bold')
            ax2.grid(True, linestyle='--', alpha=0.4)
            ax2.legend(loc='upper right', fontsize=10)
            
            # --- 子圖 3: 升阻比 ---
            front_ld = np.array(front_lift) / (np.array(front_drag) + 1e-6)
            rear_ld = np.array(rear_lift) / (np.array(rear_drag) + 1e-6)
            
            ax3.plot(steps_log, front_ld, label='Front Wing L/D', 
                    color='#2E86AB', linewidth=1.5)
            ax3.plot(steps_log, rear_ld, label='Rear Wing L/D', 
                    color='#A23B72', linestyle='--', linewidth=1.5)
            ax3.set_xlabel('Time Steps', fontsize=11)
            ax3.set_ylabel('Lift-to-Drag Ratio', fontsize=11)
            ax3.set_title('Aerodynamic Efficiency (L/D)', fontsize=12, fontweight='bold')
            ax3.grid(True, linestyle='--', alpha=0.4)
            ax3.legend(loc='upper right', fontsize=10)
            ax3.axhline(y=0, color='black', linewidth=0.5)
            
            # --- 子圖 4: 性能監控 ---
            # 移動平均以平滑曲線
            window = 50
            mlups_smooth = np.convolve(
                mlups_history, 
                np.ones(window)/window, 
                mode='valid'
            )
            ax4.plot(range(len(mlups_smooth)), mlups_smooth, 
                    color='#06A77D', linewidth=1.5)
            ax4.set_xlabel('Time Steps', fontsize=11)
            ax4.set_ylabel('Performance (MLUPS)', fontsize=11)
            ax4.set_title(f'Computational Performance (Avg: {avg_mlups:.1f} MLUPS)', 
                         fontsize=12, fontweight='bold')
            ax4.grid(True, linestyle='--', alpha=0.4)
            ax4.axhline(y=avg_mlups, color='red', linestyle='--', 
                       linewidth=1, label=f'Average: {avg_mlups:.1f}')
            ax4.legend(loc='lower right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig("tandem_force_analysis.png", dpi=300, bbox_inches='tight')
            print("✓ 科學數據圖表已保存: tandem_force_analysis.png")
            
            # ========================================
            # 保存原始數據（CSV 格式）
            # ========================================
            data_array = np.column_stack([
                steps_log, 
                front_lift, front_drag, 
                rear_lift, rear_drag,
                front_ld, rear_ld
            ])
            np.savetxt(
                "force_data.csv",
                data_array,
                delimiter=',',
                header='Step,Front_Lift,Front_Drag,Rear_Lift,Rear_Drag,Front_LD,Rear_LD',
                comments=''
            )
            print("✓ 原始數據已保存: force_data.csv")
            
        else:
            print("⚠️ 警告: 沒有記錄到數據點")
    
    except Exception as e:
        print(f"❌ 繪圖失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # 總結報告
    # ========================================
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("📊 模擬總結")
    print("=" * 70)
    print(f"  網格尺寸: {cfg.NX} × {cfg.NY} = {cfg.NX*cfg.NY/1e6:.1f}M 格點")
    print(f"  Reynolds 數: {cfg.RE:,}")
    print(f"  總步數: {cfg.MAX_STEPS:,}")
    print(f"  總時長: {format_time(total_time)}")
    print(f"  平均性能: {avg_mlups:.1f} MLUPS")
    print(f"  記憶體使用: ~{cfg.NX*cfg.NY*88/1e9:.1f} GB")
    print("=" * 70)
    print("\n✨ 所有任務完成！")
    print("輸出文件:")
    print("  - tandem_wing_simulation.mp4  (流場視覺化)")
    print("  - tandem_force_analysis.png   (受力分析圖)")
    print("  - force_data.csv              (原始數據)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
