import numpy as np, os

mask_dir = 'data/ISIC2017'
for fname in ['mask_train.npy', 'mask_val.npy', 'mask_test.npy']:
    f = os.path.join(mask_dir, fname)
    if os.path.exists(f):
        arr = np.load(f)
        print(f'{fname:>12}  shape:{arr.shape}  unique:{np.unique(arr)}')
    else:
        print(f'{fname:>12}  文件不存在！')
