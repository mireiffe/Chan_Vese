from os.path import join

import matplotlib.pyplot as plt

import numpy as np
import cv2
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes
from segmentation import ChanVese

import myTools as mts
from anisodiff import anisodiff

if __name__ == '__main__':
    N = 16
    cvseg = ChanVese(
        N=N, nu=.1, dt=2, tol=1E-03,
        method='vector', initial='smart',
        # method='vector', initial=None,
        # method='gray', initial=None,
        reinterm=10, vismode=True, visterm=10
        # reinterm=10, vismode=False, visterm=10
    )

    dir_data = '/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/validation/'
    dir_img = f'{dir_data}data/'
    dir_mask = f'{dir_data}mask/'
    dir_save = './results/'

    nm_imgs = [39769] # cat
    # nm_imgs = [59598]
    # nm_imgs = [21465] # vase
    nm_imgs = [348481] # desk
    for nm_img in nm_imgs:
        name_save = join(dir_save, f'{nm_img:012d}_{N}')
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
        img = np.stack([anisodiff(img0[..., i], niter=300, kappa=5, gamma=0.1, option=2)
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

        res = mts.loadFile(name_save + '/phis.pck')
        phis = res['phis']
        c = res['c']
        m, n = phis[0].shape
        Hs = np.stack((phis > 0, phis < 0), axis=1)
        pc_img = np.zeros_like(img).astype(float)
        for dn in range(2**len(phis)):
            bn = eval(f"f'{dn:0{len(phis)}b}'")
            _H = np.ones_like(Hs[0][0])
            for ip in range(len(phis)):
                _H = _H * Hs[ip][int(bn[ip])]
            mean_reg = (_H > .5) * (1 - mask)
            _H = _H[..., np.newaxis]
            pc_img += _H * c[dn] * (1 - mask)[..., np.newaxis]

        res_seg = np.zeros_like(pc_img)

        cny_img = np.stack([anisodiff(img0[..., i], niter=200, kappa=5, gamma=0.1, option=2)
                 for i in range(img0.shape[2])], axis=2) / 255
        cny0 = cv2.Canny((cny_img *255 * (1-mask)[..., np.newaxis]).astype('uint8'), 30, 70)
        cny = mts.imDilErod(cny0, rad=3)
        # _cny = np.ones((cny.shape[0] + 2, cny.shape[1] + 2))
        # _cny[1:-1, 1:-1] = cny
        # _fill_cny = binary_fill_holes(_cny)[1:-1, 1:-1]
        _fill_cny = binary_fill_holes(cny)
        fill_cny = mts.imErodDil(_fill_cny.astype(float), rad=4) * (1-mask)

        cal_idx = np.where(1 - mask)
        u_clr, cnt_clr = np.unique(pc_img[cal_idx], return_counts=True, axis=0)
        srt_cnt = np.argsort(cnt_clr)

        lbl_pc = np.zeros_like(phis[0])
        for uc in u_clr:
            _r = np.where((pc_img == uc).prod(axis=2), 1., 0.)
            _lbl = label(_r, connectivity=1)
            lbl_pc = np.where(_lbl > .5, _lbl + lbl_pc.max(), lbl_pc)

        res_lbl = np.zeros_like(lbl_pc)
        for l in np.unique(lbl_pc)[1:]:
            _r = np.where(lbl_pc == l, 1., 0.)
            prt = _r.sum() / m / n
            llim = 1E-04
            if ((fill_cny * _r).sum() / _r.sum() >= .5) and (_r.sum() >= llim):
                res_seg = np.where(lbl_pc[..., np.newaxis] == l, pc_img, res_seg)
                res_lbl = np.where(lbl_pc == l, lbl_pc, res_lbl)

        # for l in np.unique(lbl_pc)[1:]:
        #     _r = np.where(lbl_pc == l, 1., 0.)
        #     prt = _r.sum() / m / n
        #     llim = 5E-04
        #     if (prt > .01) or (prt < llim):
        #         _bt = (l == lbl_pc[[0, -1], :]).sum()
        #         _lr = (l == lbl_pc[:, [0, -1]]).sum()
        #         if ((_bt + _lr)**2 > _r.sum() / 20)  or (prt < llim):
        #             if (fill_cny * _r).sum() > .9 * _r.sum(): continue
        #             res_seg = np.where(lbl_pc[..., np.newaxis] == l, 0., res_seg)
        #             lbl_pc = np.where(lbl_pc == l, 0., lbl_pc)

        eig_lst = {}
        rat_lst = {}
        cenm_lst = {}
        for l in np.unique(lbl_pc)[1:]:
            r_idx = np.where(lbl_pc == l)

            # y and x order
            cenm = np.sum(r_idx, axis=1) / len(r_idx[0])
            cen_idx = r_idx[0] - cenm[0], r_idx[1] - cenm[1]

            Ixx = np.sum(cen_idx[0]**2)
            Iyy = np.sum(cen_idx[1]**2)
            Ixy = -np.sum(cen_idx[0]*cen_idx[1])

            intiaT = [[Ixx, Ixy], [Ixy, Iyy]]

            D, Q = mts.sortEig(intiaT)

            eig_lst[l] = (D, Q)
            rat_lst[l] = D[0] / D[1]
            cenm_lst[l] = cenm

        for l in np.unique(lbl_pc):
            if l <= 0: continue
            if (rat_lst[int(l)] >= 8):
                _r = np.where(lbl_pc == l, 1., 0.)
                res_seg = np.where(lbl_pc[..., np.newaxis] == l, 0., res_seg)
                lbl_pc = np.where(lbl_pc == l, 0., lbl_pc)

        plt.figure()
        plt.imshow(img0 * (mask + (res_lbl > .5))[..., np.newaxis])
        mts.savecfg(name_save + '/result.png')