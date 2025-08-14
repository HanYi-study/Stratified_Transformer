import numpy as np
import glob
import os

files = glob.glob('/home/hy/projects/Stratified_Transformer/data_prepare_output/*.npy')
for f in files:
    arr = np.load(f)
    if arr.shape[1] == 7:
        # 插入两列全0，label仍在最后一列
        arr_new = np.concatenate([arr[:, :6], np.zeros((arr.shape[0], 2)), arr[:, 6:]], axis=1)
        save_dir = '/home/hy/projects/Stratified_Transformer/data_prepare_output_converted'
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, os.path.basename(f)), arr_new)
        print(f"Converted {f} to shape {arr_new.shape}")
    else:
        print(f"Skipped {f}, shape is {arr.shape}")