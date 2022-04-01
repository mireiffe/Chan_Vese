import os
import time
import pickle

from colorama import Fore
from tqdm import tqdm
from tqdm._utils import _term_move_up

import cv2
import numpy as np
import matplotlib.pyplot as plt

import myTools as mts


class ChanVese(object):
    eps = np.finfo(float).eps

    def __init__(self, N=2, nu, mu, dt=.2, initial=None, reinterm=5, vismode=False, visterm=5):
        self.N = N
        self.n_phi = int(np.ceil(np.log2(N)))
        self.mu = mu
        self.nu = nu
        self.dt = dt
        self.tol = tol
        self.initial = initial
        self.reinterm = reinterm
        self.vismode = vismode
        self.visterm = visterm

    def initC(self):
        if self.initials == None:
            self.phis = [cv2.resize(inis, self.img.shape[::-1], interpolation=cv2.INTER_LINEAR)
                        for inis in self.initials[0]]
            self.b = cv2.resize(self.initials[1], self.img.shape[::-1], interpolation=cv2.INTER_LINEAR)
            self.c = self.initials[2]

        else:
            self.b = np.ones_like(self.img)
            self.c = [0] * self.N

            _mu, _sig = self.img.mean(), self.img.std()
            _lev = range(1, 1 - self.n_phi, -1)
            _lev = range(1, 1 + self.n_phi)
            #self.phis = [np.where(self.img > _mu + l * _sig, -1., 1.) for l in _lev]
            self.phis = [np.where(self.img > _mu + l * _sig, (-1)**k * -1., (-1)**k * 1.) for k, l in enumerate(_lev)]

    def updt_bbk(self):
        self.bk = cv2.filter2D(self.b, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        self.bbk = cv2.filter2D(self.b * self.b, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        
    def updt_bc(self):
        _oldc, _oldb = np.copy(self.b), np.copy(self.c)
        # update c
        self.H = [self.funs.hvsd(phi) for phi in self.phis]
        H_ref = [(h, 1 - h) for h in self.H]
        M = [np.prod(m, axis=0) for m in itertools.product(*H_ref)]
        c_den = [np.sum(self.bk * self.img * m) for m in M]
        c_num = [np.sum(self.bbk * m) for m in M]
        self.c = nzdiv(c_den, c_num)

        # update b
        J1 = np.sum(np.prod((self.c, M), axis=0))
        J2 = np.sum(np.prod((np.power(self.c, 2), M), axis=0))
        b_den = cv2.filter2D(self.img * J1, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        b_num = cv2.filter2D(J2, -1, self.ker[::-1, ::-1], borderType=cv2.BORDER_CONSTANT)
        self.b = nzdiv(b_den, b_num)
        _newc, _newb = np.copy(self.b), np.copy(self.c)
        self.err_b = nzdiv(self.ops.norm(_oldb - _newb), (self.ops.norm(_newb)))
        self.err_c = nzdiv(self.ops.norm(_oldc - _newc), (self.ops.norm(_newc)))

        self.err_b /= self.funs.fun_dt(self.glb_t) * self.dt
        self.err_c /= self.funs.fun_dt(self.glb_t) * self.dt

    def updt_phi(self):
        self.updt_bbk()
        
        _e = [
            self.img ** 2 * self.onek - 2 * ci * self.img * self.bk + ci ** 2 * self.bbk
            for ci in self.c
        ]
        phi_t = 0

        _pbar = tqdm(
            total=50,
            desc=f'Updating phi', unit='iter', leave=False, 
            bar_format='{l_bar}%s{bar:25}%s{r_bar}{bar:-25b}' % (Fore.BLUE, Fore.RESET)
        )
        while True:
            phi_t += 1
            self.glb_phi_t += 1

            _old = np.copy(self.phis)

            gphis = [self.ops.grad_img(phi) for phi in self.phis]
            N_gphis = [self.ops.norm(gphi) for gphi in gphis]
            
            kappa = [self.ops.cvt_phi(phi, ksz=1) for phi in self.phis]
            delta_phi = [self.funs.delta(phi) for phi in self.phis]

            dp_dpgs = [
                [self.funs.fun_dp(n_gphi) * gphi[0], self.funs.fun_dp(n_gphi) * gphi[1]]
                for n_gphi, gphi in list(zip(*[N_gphis, gphis]))
            ]
            divg = [self.ops.div_phi(ddp, ksz=1) for ddp in dp_dpgs]

            if self.N >= 1 and self.N <= 2:
                dE = [_e[0] - _e[1]]
            elif self.N >= 3 and self.N <= 4:
                dE = [self.H[1] * (_e[0] - _e[2]) + (1 - self.H[1]) * (_e[1] - _e[3]),
                    self.H[0] * (_e[0] - _e[1]) + (1 - self.H[0]) * (_e[2] - _e[3])]
            else:
                print('Use appropriate value of N!!')

            dphis = [
                - dp * de + self.nu * dp * kp + self.mu * dv
                for dp, kp, dv, de in list(zip(*[delta_phi, kappa, divg, dE]))
            ]
            self.phis = [
                phi + (self.dt * self.funs.fun_dt(self.glb_t)) * nzdiv(dphi, np.abs(dphi).max())
                for phi, dphi in list(zip(*[self.phis, dphis]))
            ]
            if phi_t == 1:
                self.phis = [np.where(phi < 0, -1., 1.) for phi in self.phis]
            
            _new = np.copy(self.phis)
            err_reg = np.where(np.abs(_new) < 1.5, 1., 0.)

            self.err_phi = [
                self.ops.norm(err_reg * (o - n)) / err_reg.sum() / self.dt / self.funs.fun_dt(self.glb_t)
                for o, n in list(zip(*[_old, _new]))
            ]

            _pbar.set_postfix_str('Error phi='+', '.join([f'{ep:.2E}' for ep in self.err_phi]))
            _pbar.update()
            
            self.guis()

            if (phi_t > 10 and np.max(self.err_phi) < self.tol[0]) or phi_t > 50:
                _pbar.close()
                break

    def guis(self, fignum=500, keep=False, keep_time=0.01, mask=None):
        if self.vismode:
            if self.visterm == 0:
                if mask is not None:
                    fig = plt.figure(fignum)
                    ax = fig.subplots(2, 1)
                    ax[0].cla()
                    ax[0].imshow(self.img, 'gray')
                    clrs = ['red', 'green']
                    print(phis[0].shape)
                    for i, phi in enumerate(self.phis):
                        ax[0].contour(phi, levels=0, colors=clrs[i], linestyles='solid')
                    ax[1].imshow(mask)
                    ax[1].imshow(self.img, 'gray', alpha=0.75)
                    if keep:
                        plt.show()
                    else:
                        plt.pause(keep_time)
            elif self.glb_phi_t % self.visterm == 0 or mask is not None:
                fig = plt.figure(fignum)
                if mask is not None:
                    fig.clf()
                    ax = fig.subplots(2, 1)
                    ax[0].cla()
                    ax[0].imshow(self.img, 'gray')
                    clrs = ['red', 'green']
                    for i, phi in enumerate(self.phis):
                        ax[0].contour(phi, levels=0, colors=clrs[i], linestyles='solid')
                    ax[1].imshow(mask)
                    ax[1].imshow(self.img, 'gray', alpha=0.75)
                    if keep:
                        plt.show()
                    else:
                        plt.pause(keep_time)
                else:
                    fig.clf()
                    ax = fig.subplots(1, 1)
                    ax.cla()
                    M1 = (self.phis[0] < 0) * (self.phis[1] < 0)
                    M2 = (self.phis[0] < 0) * (self.phis[1] >= 0)
                    M3 = (self.phis[0] >= 0) * (self.phis[1] < 0)
                    M4 = (self.phis[0] >= 0) * (self.phis[1] >= 0)
  
                    M = M1 + 2*M2 + 3*M3 + 4*M4
                    ax.imshow(M)
                    clrs = ['red', 'lime']
                    for i, phi in enumerate(self.phis):
                        ax.contour(phi, levels=0, colors=clrs[i], linewidths=1.5)
                    if keep:
                        plt.show()
                    else:
                        plt.pause(keep_time)
