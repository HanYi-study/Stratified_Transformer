import numpy as np
import os
# 用于检查官方格式最后一列是否全部都是labels
folder = 'data_prepare_output_converted'
files = [f for f in os.listdir(folder) if f.endswith('.npy')]

all_ok = True
for fname in files:
    path = os.path.join(folder, fname)
    data = np.load(path)
    last_col = data[:, -1]
    if not np.all(np.equal(np.mod(last_col, 1), 0)):
        print(f"{fname} 不是全为整数标签")
        all_ok = False
        break

if all_ok:
    print(True)