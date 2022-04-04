from os.path import join

import cv2
import numpy as np
import matplotlib.pyplot as plt

from segmentation import ChanVese
import myTools as mts


if __name__ == '__main__':
    cvseg = ChanVese(
        N=8, nu=.1, dt=.3, tol=1E-03,
        method='vector', initial=None,
        # method='gray', initial=None,
        reinterm=10, vismode=True, visterm=50
        # reinterm=10, vismode=False, visterm=10
    )

    dir_save = './results/'
    nm_imgs = ['000000046048']
    nm_imgs = ['000000039769']

    def quantimage(image,k):
        i = np.float32(image).reshape(-1,3)
        condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
        ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        final_img = center[label.flatten()]
        final_img = final_img.reshape(image.shape)
        return final_img

    from skimage.segmentation import slic
    from skimage.measure import label
    from skimage.segmentation import mark_boundaries
    from anisodiff import anisodiff

    for nm_img in nm_imgs:
        try:
            img0 = plt.imread(f'./data/{nm_img}.jpg')
        except FileExistsError:
            img0 = plt.imread(f'./data/{nm_img}.png')
        img = mts.gaussfilt(img0, sig=1)
        aimg = anisodiff(img0, niter=5)

        sts = mts.SaveTools(join(dir_save, nm_img))
        # pc_img, phis = cvseg.segmentation(img)

        superpixel = slic(img0, n_segments = 500, sigma = 1, start_label=1)
        quant = quantimage(img0, 16)
        plt.figure(); plt.imshow(mark_boundaries(quant, superpixel))

        pass
        # mts.makeDir(join(dir_save, nm_img))
        # sts.imshow(img0, 'input')
        # sts.imshow(pc_img / 255, 'output')
        # sts.saveFile({'img': img0, 'phis': phis}, 'phis.pck')