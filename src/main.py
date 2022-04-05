from os.path import join

import matplotlib.pyplot as plt

from anisodiff import anisodiff
from segmentation import ChanVese
import myTools as mts
from gadf import GADF


if __name__ == '__main__':
    cvseg = ChanVese(
        N=4, nu=5, dt=.3, tol=1E-03,
        method='vector', initial=None,
        # method='gray', initial=None,
        reinterm=10, vismode=True, visterm=10
        # reinterm=10, vismode=False, visterm=10
    )

    dir_img = '/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/val2017/'
    dir_save = './results/'
    nm_imgs = ['000000046048']
    nm_imgs = ['000000039769']
    nm_imgs = ['000000547383']

    for nm_img in nm_imgs:
        try:
            img0 = plt.imread(f'{dir_img}{nm_img}.jpg')
        except FileExistsError:
            img0 = plt.imread(f'{dir_img}{nm_img}.png')
        img = mts.gaussfilt(img0, sig=2)
        aimg = anisodiff(img0, niter=20)
        sts = mts.SaveTools(join(dir_save, nm_img))

        gadf = GADF(aimg, sig=0)
        fa = gadf.Fa
        er = gadf.Er


        pass
        # mts.makeDir(join(dir_save, nm_img))
        # sts.imshow(img0, 'input')
        # sts.imshow(pc_img / 255, 'output')
        # sts.saveFile({'img': img0, 'phis': phis}, 'phis.pck')
