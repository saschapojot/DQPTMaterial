import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool

import scipy.linalg as slin
from scipy import integrate as integrate


#this script contains functions for nonhermitian case

#consts
mu=0.6
r=0.3
Gamma=1
cutOff=0.8*np.pi
#########
def h1(k):
    return mu+r*np.cos(k)
def h3(k):
    return r*np.sin(k)


def psi0(k):
    """

    :param k:
    :return: init vector
    """
    h1Val=h1(k)
    h3Val=h3(k)
    E=np.sqrt(h1Val**2+h3Val**2)
    denomTmp=np.sqrt(2*E**2+2*h3Val*E)
    rst=[-h1Val/denomTmp,(h3Val+E)/denomTmp]
    return np.array(rst)

def Hf(k):
    H = np.zeros((2, 2), dtype=complex)
    H[0, 0] = h3(k) + 1j * 1 / 2 * Gamma
    H[0, 1] = h1(k)
    H[1, 0] = h1(k)
    H[1, 1] = -h3(k) - 1j * 1 / 2 * Gamma

    return H

def expHkF(k,t):
    """U=expHkF(1,2)

    :param k:
    :param t:
    :return: exp(-iH_{k}^{f}t)
    """


    U = slin.expm(-1j * t * Hf(k))

    return U

def psikt(k,t):
    """

    :param k:
    :param t:
    :return:
    """
    return expHkF(k, t).dot(psi0(k))

def integrand(k,t):
    '''

    :param k:
    :param t:
    :return: integrand to calculate beta
    '''
    psikt=expHkF(k,t).dot(psi0(k))
    HfMat=Hf(k)
    N=psikt.conj().T@HfMat@psikt
    D=psikt.conj().T@psikt

    return N/D

def integral(k,tStart,tEnd):
    """

    :param k:print(integral(1,2,3))
    :param tStart:
    :param tEnd:
    :return:
    """
    func=lambda t:integrand(k,t)
    reFunc=lambda t:np.real(func(t))
    imFunc=lambda t:np.imag(func(t))
    return integrate.quad(reFunc,tStart,tEnd)[0]+1j*integrate.quad(imFunc,tStart,tEnd)[0]


def G(k,t):
    psi0Vec=psi0(k)
    psiktVec=psikt(k,t)
    return psi0Vec.conj().T.dot(psiktVec)






def jump(dTheta):
    if dTheta>=cutOff:
        while dTheta>=cutOff:
            dTheta-=2*np.pi
    elif dTheta<=-cutOff:
        while dTheta<=-cutOff:
            dTheta+=2*np.pi

    return dTheta
