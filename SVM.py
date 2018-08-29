import numpy as np
import matplotlib.pyplot as plt

# 加载文件，读取数据
def load_data(fileName):
    dataMat = []
    labelMat = []
    file = open(fileName)
    for line in file.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 生成不等于i的随机数
def selectJrand(i,m):
    j=i 
    while (j==i):
        j=int(np.random.uniform(0,m))
    return j

# 限定aj的取值范围
def clipAlpha(aj,H,L):
    if aj>H:
        aj=H 
    elif aj<L:
        aj=L 
    return aj

# 核函数
def kernelTrans(X,A,kTup):
    m,n=np.shape(X)
    k=np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        k=X*A.T 
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            k[j]=np.exp(deltaRow*deltaRow.T/(-1*kTup[1]**2))
    elif kTup[0]=='poly':
        for j in range(m):
            k[j]=(X[j,:]*A.T+1)**kTup[1]
    return k

# SMO类参数
class optStruct:
    def __init__(self,dataMat,labelMat,C,toler,kTup):
        self.X=dataMat 
        self.labelMat=labelMat 
        self.C=C 
        self.tol=toler 
        self.m=np.shape(dataMat)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0 
        self.eCache=np.mat(np.zeros((self.m,2)))
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],kTup)

# 计算损失函数
def calcEk(oS,k):
    fxk=float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k])+oS.b 
    Ek=fxk-float(oS.labelMat[k])
    return Ek
             
# 选择第二个alpha
def selectJ(i, oS, Ei):         
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:    
            if k == i: continue  
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:    
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 计算误差值并存入缓存中
def updateEk(oS, k): 
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

# SMO算法的内循环
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei)  
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] 
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j)  
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j]) 
        updateEk(oS, i)                
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

# SMO算法的外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:    
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("FullSet iter=%d,  i=%d,  alphaPairsChanged=%d" % (iter,i,alphaPairsChanged))
            iter += 1
        else: 
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("Non-bound iter=%d,  i=%d,  alphaPairsChanged=%d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False  
        elif (alphaPairsChanged == 0): entireSet = True  
        print("Iteration number=%d" % iter)
    return oS.b,oS.alphas

# 计算超平面向量w
def calcWs(alphas,dataArr,classLabels):
    x = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(x)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i],x[i,:].T)
    return w

# 调用SMO函数,计算参数b和alphas,以及超平面向量w
dataMat,labelMat = load_data('test_data.txt')
b,alphas = smoP(dataMat,labelMat,C=20,toler=0.001,maxIter=1000)
w = calcWs(alphas,dataMat,labelMat)

# 检验正确率
x = np.mat(dataMat)
y = np.mat(labelMat)
m,n = np.shape(x)
errorCount = 0
for i in range(m):
    predict = x[i,:] * w + b
    if np.sign(predict) != np.sign(y[i]):
        errorCount += 1
print("Training error rate = %f" % (float(errorCount)/m))

# 训练样本画图
x = np.array(x); y = np.array(y)
lp_x1 = x[:,0]
lp_x2 = x[:,1]
xmin, xmax = min(lp_x1)-1, max(lp_x1)+1
ymin, ymax = min(lp_x2)-1, max(lp_x2)+1
for i in range(0,len(x)):
    if y[i] == 1:
        plt.scatter(lp_x1[i], lp_x2[i], marker='s', s=150, c='blue')
    elif y[i] == -1:
        plt.scatter(lp_x1[i], lp_x2[i], marker='o', s=150, c='red')

# 画超平面直线
x = np.arange(xmin, xmax, 0.1)
y = np.array((-w[0] * x - b)/ w[1])
y = y[0]
plt.plot(x,y)
plt.show()