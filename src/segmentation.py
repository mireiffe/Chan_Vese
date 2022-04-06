import os
import time
import pickle

from tqdm import tqdm
from colorama import Fore
from tqdm._utils import _term_move_up

import cv2
import numpy as np
import matplotlib.pyplot as plt

import myTools as mts
from reinitial import Reinitial


class ChanVese(object):
    eps = np.finfo(float).eps

    def __init__(self, N=2, nu=.5, dt=.2, method='gray', initial=None, tol=1E-03, reinterm=5, vismode=False, visterm=5):
        self.N = N
        self.nu = nu
        self.dt = dt
        self.tol = tol
        self.method = method
        self.initial = initial
        self.reinterm = reinterm
        self.vismode = vismode
        self.visterm = visterm

        self.n_phi = int(np.ceil(np.log2(N)))

    def segmentation(self, img: np.ndarray, mask: np.ndarray):
        img0 = img
        keepReg = 1 - mask
        if img.ndim == 1:
            assert 'Image dimension must be larger than 2'
        elif img.ndim == 2:
            if self.method == 'vector':
                img = img[..., np.newaxis]
        elif img.ndim >= 3:
            if self.method == 'gray':
                img = img.mean(axis=2)

        rein = Reinitial(dim_stack=0)
        self.phis0 = self.initC(img, rein)
        self.phis = np.copy(self.phis0)
        k = 0
        _, (ax0, ax1) = plt.subplots(1, 2)
        while True:
            Hs = np.stack((mts.hvsd(self.phis), mts.hvsd(-self.phis)), axis=1)
            c, pc_img = self.mkC(img, Hs, keepReg)
            kapps = mts.kappa(self.phis, mode=0, stackdim=0)[0]
            dE = self.mkDE(img, Hs, c, keepReg)

            # _phis = self.phis + self.dt * (self.nu * kapps - (dE - 2*mask) * mts.delta(self.phis))
            _phis = self.phis + self.dt * (self.nu * kapps - (dE - 1.5*mask))

            print(f"Iteration: {k:d}", end='\r')
            if self.vismode and (k % self.visterm == 0):
                ax0.cla()
                ax1.cla()

                #axis0
                ax0.imshow(img0)
                clrs = ['lime', 'red', 'blue', 'yellow']
                for i, ph in enumerate(_phis):
                    ax0.contour(ph, levels=[0], colors=clrs[i], linewidths=1.5)
                ax0.set_title(f'Method: {self.method}')

                # axis1
                try:
                    ax1.imshow(pc_img, 'gray')
                except:
                    ax1.imshow(pc_img[..., 0], 'gray')
                ax1.set_title(f'Iter: {k:d}')
                plt.pause(0.05)
            
            if k % self.reinterm == 0:
                _phis = rein.getSDF(np.where(_phis < 0, -1., 1.))
                # _phis = rein.getSDF(_phis)

            err = np.sqrt(((_phis - self.phis)**2).sum()) / np.sqrt(((_phis)**2).sum())
            if (k >= 3000) or err / self.dt < self.tol:
                break

            self.phis = np.copy(_phis)
            k += 1
        return pc_img, _phis
        
    def initC(self, img: np.ndarray, rein):
        shifts = [(0, 0), (1.75, 1), (1, 1.75), (1.35, 1.35)]
        # shifts = [(0, 0)] * 4
        nums = [20, 20, 20, 20]
        if self.initial == None:
            m, n = img.shape[:2] 
            circs = np.array([mts.patCirc(m, n, nums=nums[_], shift=shifts[_]) for _ in range(self.n_phi)])
        return rein.getSDF(np.where(circs, -1., 1.))

    def mkC(self, img, Hs, keepReg):
        c = []
        pc_img = np.zeros_like(img).astype(float)
        for dn in range(2**self.n_phi):
            bn = eval(f"f'{dn:0{self.n_phi}b}'")
            _H = np.ones_like(Hs[0][0])
            for ip in range(self.n_phi):
                _H = _H * Hs[ip][int(bn[ip])]
            mean_reg = (_H > .5) * keepReg
            if self.method == 'gray':
                c.append((img * mean_reg).sum(axis=(0, 1)) / ((mean_reg).sum() + 1E-05))
            elif self.method == 'vector':
                c.append((img * mean_reg[..., np.newaxis]).sum(axis=(0, 1)) / (mean_reg.sum() + 1E-05))
                _H = _H[..., np.newaxis]
            pc_img += _H * c[dn] * keepReg[..., np.newaxis]
        return np.array(c), pc_img

    def mkDE(self, img, Hs, c, keepReg):
        m, n = img.shape[:2]
        de = []
        for j, phi in enumerate(self.phis):
            _d = np.zeros_like(phi)
            for t in range(2**self.n_phi):
                bn = eval(f"f'{t:0{self.n_phi}b}'")
                _r = np.ones_like(_d)
                for i in np.setdiff1d(range(self.n_phi), j):
                    _r = _r * Hs[i][int(bn[i])]
                _rr = Hs[j][int(bn[j])]
                _e1 = ((img - c[t])**2 * (_rr * _r)[..., np.newaxis]).sum(axis=(0, 1))
                _e = ((img - c[t])**2 - _e1 / (_rr * _r).sum())* keepReg[..., np.newaxis]  / (_rr * _r).sum()
                if self.method == 'vector':
                    _e = _e.sum(axis=2)
                _d += (-1)**(int(bn[j]) + 0) * _e * _r
            de.append(_d)
        return np.array(de)