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

    def quantimage(image,k):
        i = np.float32(image).reshape(-1,3)
        condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
        ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        final_img = center[label.flatten()]
        final_img = final_img.reshape(image.shape)
        return final_img

    dir_img = '/home/users/mireiffe/Documents/Python/Pose2Seg/downloads/coco2017/validation/data'
    dir_save = './results/'
    nm_imgs = ['000000046048']
    nm_imgs = ['000000039769']
    # nm_imgs = ['000000059598']
    for nm_img in nm_imgs:
        try:
            img0 = plt.imread(f'{dir_img}{nm_img}.jpg')
        except FileExistsError:
            img0 = plt.imread(f'{dir_img}{nm_img}.png')
        img = mts.gaussfilt(img0, sig=1.5)

        sts = mts.SaveTools(join(dir_save, nm_img))
        quant_img = quantimage(img0,8)

        plt.figure()
        plt.imshow(quant_img)

        pass

        mts.makeDir(join(dir_save, nm_img))
        sts.imshow(img0, 'input')
        sts.imshow(quant_img / 255, 'output')