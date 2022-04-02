import matplotlib.pyplot as plt

from segmentation import ChanVese
import myTools as mts


if __name__ == '__main__':
    cvseg = ChanVese(
        N=16, nu=.1, dt=.2, tol=1E-03,
        method='vector', initial=None,
        # method='gray', initial=None,
        reinterm=10, vismode=True, visterm=10
    )

    # img = plt.imread('./data/4colors.jpg')
    img = plt.imread('./data/flowers.jpg')
    img = mts.gaussfilt(img, sig=1)

    cvseg.segmentation(img)