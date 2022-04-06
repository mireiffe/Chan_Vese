from os.path import join

import matplotlib.pyplot as plt

import numpy as np
from segmentation import ChanVese

import myTools as mts
from anisodiff import anisodiff

if __name__ == '__main__':
    cvseg = ChanVese(
        N=4, nu=.1, dt=.3, tol=1E-03,
        method='vector', initial=None,
        # method='gray', initial=None,
        reinterm=10, vismode=True, visterm=20
        # reinterm=10, vismode=False, visterm=10
    )

    dir_data = '/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/validation/'
    dir_img = f'{dir_data}data/'
    dir_mask = f'{dir_data}mask/'
    dir_save = './results/'

    nm_imgs = [39769]
    # nm_imgs = [59598]
    for nm_img in nm_imgs:
        name_save = join(dir_save, f'{nm_img:012d}')
        try:
            img0 = plt.imread(f'{dir_img}{nm_img:012d}.jpg')
        except FileExistsError:
            img0 = plt.imread(f'{dir_img}{nm_img:012d}.png')
        mask0 = mts.loadFile(f'{dir_mask}{nm_img:012d}.pck')
        
        # img = mts.gaussfilt(img0, sig=.5) / 255
        img = np.stack([anisodiff(img0[..., i] / 255, niter=15) for i in range(img0.shape[2])], axis=2)
        mask = mask0 > .5

        sts = mts.SaveTools(name_save)
        pc_img, phis = cvseg.segmentation(img, mask=mask)

        mts.makeDir(name_save)
        sts.imshow(img0, 'input')
        sts.imshow(pc_img / 255, 'output')
        sts.saveFile({'img': img0, 'phis': phis}, 'phis.pck')
