from funcsNonlinear import *



threadNum=24
pool1=Pool(threadNum)
vecTimeStart=datetime.now()
retAll=pool1.map(oneRow,range(0,M+1))

vecTimeEnd=datetime.now()
print("vec computation time: ",vecTimeEnd-vecTimeStart)
vecTab=[]
for m in range(0,M+1):
    vecTab.append([])

for listTmp in retAll:
    kNumTmp=listTmp[0]
    vecTmp=listTmp[1]
    vecTab[kNumTmp].extend(vecTmp)

def computeOnefkj(pairkNumj):
    kNum, j=pairkNumj
    return [kNum,j,fkj(kNum,j,vecTab)]

pairkNumjAll=[]
for kNum in range(0,M+1):
    for j in range(0,Q*N+1):
        pairkNumjAll.append([kNum,j])

pool2=Pool(threadNum)
fTabStart=datetime.now()
retkjfkj=pool2.map(computeOnefkj,pairkNumjAll)
fTabEnd=datetime.now()

print("f tab time:",fTabEnd-fTabStart)
fTab=np.zeros((M+1,Q*N+1),dtype=float)
for itemTmp in retkjfkj:
    kNum,j,fTmp=itemTmp
    fTab[kNum,j]=np.real(fTmp)

intTab=np.zeros((M+1,Q),dtype=float)#q=0,1,...,Q-1
def intTabFillOneEntry(pairkNumq):
    kNum,q=pairkNumq
    return [kNum,q,integral(kNum,q,fTab)]
pairkNumqAll=[]
for kNum in range(0,M+1):
    for q in range(0,Q):
        pairkNumqAll.append([kNum,q])

pool3=Pool(threadNum)
retkNumqInt=pool3.map(intTabFillOneEntry,pairkNumqAll)
for itemTmp in retkNumqInt:
    kNum,q,intTmp=itemTmp
    intTab[kNum,q]=intTmp

betaTab=np.zeros((M+1,Q+1),dtype=float)
for kNum in range(0,M+1):
    for q in range(0,Q):
        betaTab[kNum,q+1]=betaTab[kNum,q]+intTab[kNum,q]

alphaTab=np.zeros((M+1,Q+1),dtype=float)

for kNum in range(0,M+1):
    for q in range(0,Q+1):
        alphaTab[kNum,q]=alpha(kNum,q,vecTab)


thetaTab=alphaTab+betaTab
dThetaTab=np.zeros((M,Q+1),dtype=float)
for q in range(0,Q+1):
    for m in range(0,M):
        dThetaTmp=thetaTab[m+1,q]-thetaTab[m,q]
        dThetaTab[m,q]=jump(dThetaTmp)

sumdTheta=dThetaTab.sum(axis=0)
nuD=[elem/(2*np.pi) for elem in sumdTheta]
tValsAll=[q*N*ds for q in range(0,Q+1)]
plt.plot(tValsAll,nuD,color="black")
plt.savefig("tmpNonlin.png")