from os.path import join

import matplotlib.pyplot as plt

import numpy as np
from skimage.measure import label
from segmentation import ChanVese

import myTools as mts
from anisodiff import anisodiff

if __name__ == '__main__':
    cvseg = ChanVese(
        N=4, nu=.1, dt=2, tol=1E-03,
        method='vector', initial='smart',
        # method='vector', initial=None,
        # method='gray', initial=None,
        reinterm=10, vismode=True, visterm=30
        # reinterm=10, vismode=False, visterm=10
    )

    dir_data = '/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/validation/'
    dir_img = f'{dir_data}data/'
    dir_mask = f'{dir_data}mask/'
    dir_save = './results/'

    nm_imgs = [39769] # cat
    # nm_imgs = [59598]
    nm_imgs = [168458]
    for nm_img in nm_imgs:
        name_save = join(dir_save, f'{nm_img:012d}')
        try:
            img0 = plt.imread(f'{dir_img}{nm_img:012d}.jpg')
        except FileNotFoundError:
            img0 = plt.imread(f'{dir_img}{nm_img:012d}.png')
        try:
            mask0 = mts.loadFile(f'{dir_mask}{nm_img:012d}.pck')
        except FileNotFoundError:
            from genmask import genMask
            genMask(nm_img) 
            mask0 = mts.loadFile(f'{dir_mask}{nm_img:012d}.pck')
        
        # img = mts.gaussfilt(img0, sig=.5) / 255
        # img = np.stack([anisodiff(img0[..., i], niter=30, kappa=20, gamma=0.1, option=2)
        #          for i in range(img0.shape[2])], axis=2) / 255
        img = np.stack([anisodiff(img0[..., i], niter=500, kappa=7, gamma=0.1, option=2)
                 for i in range(img0.shape[2])], axis=2) / 255
        mask = mask0 > .5

        sts = mts.SaveTools(name_save)
        pc_img, phis, c, pc_img0 = cvseg.segmentation(img, mask=mask)

        mts.makeDir(name_save)
        sts.imshow(img0, 'img')
        sts.imshow(img, 'input')
        # sts.imshow(np.where(mask[..., np.newaxis], img0 / 255, pc_img), 'output')
        sts.imshow(pc_img, 'output')
        sts.imshow(pc_img0, 'initial')
        sts.saveFile({'img': img0, 'phis': phis, 'c': c}, 'phis.pck')

        gimg = pc_img.mean(axis=2)
        lbl = label(gimg, background=0)