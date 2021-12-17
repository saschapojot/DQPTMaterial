import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy.linalg as slin
from datetime import datetime
######consts
mu=0.6
r=0.3
N=20 #small time grid
Q=100#large time grid
tTot=20
g=10
M=100#kNum
kValsAll=[2*np.pi/M*m for m in range(0,M+1)]
ds=tTot/(N*Q)
cutOff=0.5
######################

#coefs of initial Hamiltonian
def a1(k):
    return mu+r*np.cos(k)

def a2(k):
    return 0

def a3(k):
    return r*np.sin(k)


###################
#coefs of linear part of nonlinear Hamiltonian
def b1(k):
    return a1(k)

def b2(k):
    return a2(k)

def b3(k):
    return a3(k)


def psi0(k):
    """

    :param k:
    :return: initial lower eigenvector
    """
    a1Val=a1(k)
    a2Val=a2(k)
    a3Val=a3(k)
    E=np.sqrt(a1Val**2+a2Val**2+a3Val**2)
    denom=np.sqrt(2*E**2+2*a3Val*E)
    return np.array([(1j*a2Val-a1Val)/denom,(a3Val+E)/denom])


def H0(k):
    '''

    :param k:
    :return: linear part of nonlinear Hamiltonian
    '''

    b1Val=b1(k)
    b2Val=b2(k)
    b3Val=b3(k)

    rst=[[b3Val,b1Val-1j*b2Val],[b1Val+1j*b2Val,-b3Val]]
    return np.matrix(rst)


H0MatsALl=[H0(kTmp) for kTmp in kValsAll]
expH0All=[slin.expm(-1j*H0MatTmp*ds) for H0MatTmp in H0MatsALl]

def oneStepRK2(kNum,psij):
    """

    :param kNum:
    :param psij: input vector, is np.array
    :return:
    """

    xj=psij[0]
    yj=psij[1]

    x1j=xj*np.exp(-1j*g*np.abs(xj)**2*ds/2)
    y1j=yj*np.exp(-1j*g*np.abs(yj)**2*ds/2)

    vec1=np.array([x1j,y1j])
    vec2=expH0All[kNum]@vec1
    x2j=vec2[0]
    y2j=vec2[1]

    xjp1=x2j*np.exp(-1j*g*np.abs(x2j)**2*ds/2)
    yjp1=y2j*np.exp(-1j*g*np.abs(y2j)**2*ds/2)

    return np.array([xjp1,yjp1])



def oneRow(kNum):
    """

    :param kNum: row num
    :return: dict of list of vectors with kNumth momentum
    """
    rst=[psi0(kValsAll[kNum])]#0th
    for j in range(0,N*Q):
        vecj=rst[j]
        vecNext=oneStepRK2(kNum,vecj)
        rst.append(vecNext)

    return [kNum,rst]

def fkj(kNum,j,vecTab):
 """

 :param kNum:
 :param j: small time step
 :param vecTab: table (list of list)containing all eigenvectors
 :return:
 """
 vec=vecTab[kNum][j]

 rst=(vec.conj().T@H0MatsALl[kNum]@vec)[0,0]+g*np.abs(vec[0])**4+g*np.abs(vec[1])**4
 return rst

def alpha(kNum,q,vecTab):
    """

    :param kNum:
    :param q: large time step
    :param vecTab:  table (list of list)containing all eigenvectors
    :return:
    """
    psi0=vecTab[kNum][0]
    psiTmp=vecTab[kNum][q*N]
    GTmp=psi0.conj().T.dot(psiTmp)
    return np.angle(GTmp)

def integral(kNum,q,fTab):
    """

    :param kNum:
    :param q:
    :param fTab:
    :return:
    """
    
    fVecTmp=fTab[kNum][q*N:(q*N+N+1)]
    fVecTmpOdd=[fVecTmp[j] for j in range(1,N,2)]
    fVecTmpEven=[fVecTmp[j] for j in range(2,N,2)]
    return 1/3*ds*(fVecTmp[0]+4*sum(fVecTmpOdd)+2*sum(fVecTmpEven)+fVecTmp[N])



def jump(dTheta):
    if dTheta>=cutOff:
        while dTheta>=cutOff:
            dTheta-=2*np.pi
    elif dTheta<=-cutOff:
        while dTheta<=-cutOff:
            dTheta+=2*np.pi

    return dTheta
