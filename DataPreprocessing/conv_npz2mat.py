from scipy.io import savemat
import numpy as np
import glob
import os
# npzFiles = glob.glob("../data/**/*.npz", recursive=True)
npzFiles = glob.glob("../data/multi-robot-data-3-3/*.npz", recursive=True)

for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f)
    savemat(fm, d)
    print('generated ', fm, 'from', f)
