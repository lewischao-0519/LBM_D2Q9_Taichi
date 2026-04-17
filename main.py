# main.py - 執行與錄製影片
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import config as cfg
from geometry import DomainManager
from engine import (f, f_new, init_fields, lbm_step_kernel, 
                    set_inlet_kernel, compute_macro_kernel, 
                    ux_field, uy_field)

def main():
    # 1. 初始化
    cfg.init_constants()
    init_fields()
    
    domain = DomainManager(cfg.NX, cfg.NY)
    domain.add_cylinder(cx=325, cy=170, r=38)
    domain.upload() # 將障礙物搬進 GPU
    
    # 2. 影片設定 (Kaggle 環境必備)
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(np.zeros((cfg.NY, cfg.NX)), cmap='jet', vmin=0, vmax=0.12)
    plt.colorbar(img, label='Velocity Magnitude')
    
    writer = animation.FFMpegWriter(fps=30)
    
    print("正在 GPU 上執行模擬並錄製影片...")

    with writer.saving(fig, "lbm_taichi_result.mp4", dpi=100):
        for step in range(cfg.MAX_STEPS):
            # 設定邊界擾動
            perturb = 0.005 * np.sin(step * 0.1) if step < 1000 else 0.0
            set_inlet_kernel(cfg.U_MAX, perturb)
            
            # 核心演化
            lbm_step_kernel(domain.obstacle, cfg.OMEGA)
            
            # 交換指標 (f 與 f_new 對調)
            f.copy_from(f_new) 
            
            # 每 50 步抓取一個影格
            if step % 50 == 0:
                compute_macro_kernel(domain.obstacle)
                # 從 GPU 拉回數據到 CPU 繪圖
                u_mag = np.sqrt(ux_field.to_numpy()**2 + uy_field.to_numpy()**2)
                img.set_array(u_mag)
                ax.set_title(f"Step: {step}")
                writer.grab_frame()
                if step % 500 == 0: print(f"進度: {step}/{cfg.MAX_STEPS}")

    print("完成！影片已儲存為 lbm_taichi_result.mp4")

if __name__ == "__main__":
    main()