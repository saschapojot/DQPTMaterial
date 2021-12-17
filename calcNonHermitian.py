from funcsNonHermitian import *



M=200
Q=200
tTot=20
thetaGMat=np.zeros((M+1,Q+1),dtype=complex)
alphaMat=np.zeros((M+1,Q+1),dtype=complex)
betaMat=np.zeros((M+1,Q+1),dtype=complex)
kValsAll=[2*np.pi*m/M for m in range(0,M+1)]
tValsAll=[tTot/Q*q for q in range(0,Q+1)]
tStart=datetime.now()
#construct alpha matrix
for m in range(0,M+1):
    for q in range(0,Q+1):
        kValTmp=kValsAll[m]
        tValTmp=tValsAll[q]
        psiktVec=psikt(kValTmp,tValTmp)
        psi0Vec=psi0(kValTmp)
        alphaMat[m,q]=np.angle(G(kValTmp,tValTmp))-1j*1/2*np.log(
            (psiktVec.conj().T.dot(psiktVec))/(psi0Vec.conj().T.dot(psi0Vec))
        )

#construct beta matrix
for m in range(0,M+1):
    for q in range(0,Q):
        betaMat[m,q+1]=betaMat[m,q]+integral(kValsAll[m],tValsAll[q],tValsAll[q+1])


#construct thetaG mat
for m in range(0,M+1):
    for q in range(0,Q+1):
        thetaGMat[m,q]=alphaMat[m,q]+betaMat[m,q]


dThetaMat=np.zeros((M,Q+1),dtype=float)
for q in range(0,Q+1):
    for m in range(0,M):
        dThetaTmp=thetaGMat[m+1,q]-thetaGMat[m,q]

        dThetaTmp=jump(np.real(dThetaTmp))

        dThetaMat[m,q]=dThetaTmp




sumDTheta=dThetaMat.sum(axis=0)
nuD=[elem/(2*np.pi) for elem in sumDTheta]
tEnd=datetime.now()
print("calculation time: ",tEnd-tStart)
plt.figure()
plt.plot(tValsAll,nuD,color="black")
plt.savefig("tmp.png")