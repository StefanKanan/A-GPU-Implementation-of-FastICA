import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def euclid(x):
    sum = 0
    for i in x:
        sum += i**2
    return np.sqrt(sum)

def norm(X):
    l = len(np.shape(X))
    if l == 1:
        return euclid(X)

def PCA(data, firstEig=None, lastEig=None):
    if lastEig is None:
        lastEig = len(data) - 1
    if firstEig is None:
        firstEig = 0

    OldDim = len(data) #Amount of components

    COV = np.cov(data) # Dim x Dim
    D, E = np.linalg.eig(COV)

    rankTolerance = 1e-7
    maxLastEig = np.sum(D > rankTolerance)
    if maxLastEig == 0:
        raise Exception('Eigenvalues of the covariance matrix are all smaller than tolerance \
        Please make sure that your data matrix contains nonzero values. \
        \nIf the values are very small, try rescaling the data matrix.\n')

    maxLastEig = maxLastEig - 1 #for index
    eigenvalues = np.sort(D)[::-1]

    if lastEig > maxLastEig:
        lastEig = maxLastEig

    if lastEig < OldDim-1: #if lastEig essentially changes
        lowerLimitValue = (eigenvalues[lastEig] + eigenvalues[lastEig + 1]) / 2
    else:
        lowerLimitValue = eigenvalues[OldDim - 1] - 1 #It isn't inclusive

    lowerColumns = D > lowerLimitValue

    if firstEig > 0:
        higherLimitValue = (eigenvalues[firstEig - 1] + eigenvalues[firstEig]) / 2
    else:
        higherLimitValue = eigenvalues[0] + 1 #It isn't inclusive

    higherColumns = D < higherLimitValue

    selectedColumns = lowerColumns & higherColumns

    E = E[:, selectedColumns]
    D = np.diag(D[selectedColumns]) #Eigenvalues

    return E, D

def FastICASymm(X, whitening, dewhitening, maxIterations, threshold):
    Dim = len(X)
    NumOfSampl = len(X[0])

    B = linalg.orth(np.random.random((Dim, Dim)))
    Bold = np.zeros((Dim, Dim))
    #W
    A = np.zeros((Dim, Dim)) #maybe dtype

    #helpers
    sqrt = linalg.sqrtm  # sqrt on a matrix
    inv = np.linalg.inv
    
    for i in range(0, maxIterations + 1):
        if i == maxIterations:
            print('Component {} did not converge after {} iterations'.format(i, maxIterations))
            if B.size != 0: #not empty
                B = B @ np.real(inv(sqrt(B.T @ B)))
                W = B.T @ whitening
                A = dewhitening @ B

                return A, W
            return None, None #TODO
#         print(i)
        B = B @ np.real(inv(sqrt(B.T @ B))) #todo theory

        minAbsCos = min(abs(np.diag(B.T @ Bold)))
        if 1 - minAbsCos < threshold:
            pass
#             print('Converged!') #TODO
            #A = dewhitening @ B
            #W = B.T @ whitening
            #return A, W

        Bold = B

        hypTan = np.tanh(X.T @ B)
        left = (X @ hypTan) / NumOfSampl
        rowSum = np.sum(1 - hypTan ** 2, axis=0)
        #####
        right = (np.ones((Dim, Dim)) * rowSum) * B / NumOfSampl
        B = left - right

#todo
def FastICADefl(X, whitening, dewhitening, maxIterations, threshold):
    Dim = len(X)
    NumOfSampl = len(X[0])
    
    if not np.isreal(X).all():
        raise Exception('X has imaginary values!')

    B = np.zeros((Dim, Dim)) #maybe dtype
    W = np.zeros((Dim, Dim)) #maybe dtype
    A = np.zeros((Dim, Dim)) #maybe dtype
    failureLimit = 5
    numOfFailures = 0
    
    for i in range(0, Dim):
        w = np.random.random(Dim)

        w = w - B @ B.T @ w #w is a column with Dim rows
        w = w/norm(w)

        wOld = np.zeros(Dim)

        for j in range(0, maxIterations + 1):
            w = w - B@B.T@w #w is a column with Dim rows todo
            w = w/norm(w)

            if j == maxIterations + 1:
                print('Component {} did not converge after {} iterations'.format(i, maxIterations))
                numOfFailures += 1
                if numOfFailures > failureLimit: #limit is set to 5
                    raise Exception('Exceeded number of convergence failures!')

            if norm(w - wOld) < threshold or norm(w + wOld) < threshold:
                numOfFailures = 0 #?
                B[:, i] = w
                A[:, i] = dewhitening @ w
                W[i, :] = w.T @ whitening
                break

            wOld = w

            hypTan = np.tanh(X.T @ w)
            left = (X @ hypTan) / NumOfSampl
            rowSum = np.sum(1 - hypTan ** 2, axis=0)
            #####
            right = rowSum * w / NumOfSampl
            w = left - right

            w = w/norm(w)

    return A, W

#Using g(y) = tanh(y)
def FastICA(mixedsig, approach='Symm', maxIterations=10, threshold=0.0001):
    Dim = len(mixedsig)

    #Center X
    m = np.mean(mixedsig, 1)
    m = m.reshape((Dim, 1))

    mixedmean = mixedsig.copy() - m

    #PCA and whitening
    #E - the eigenvectors, an orthogonal matrix, Dim x Dim-k
    #D - the eigenvalues
    E, D = PCA(mixedsig) #maybe check if D contains negative values
    #todo read theory about mixedmin

    WhiteningMatrix = np.linalg.inv(np.sqrt(D)) @ E.T
    DewhiteningMatrix = E @ np.sqrt(D)

    Whitesig = WhiteningMatrix @ mixedsig

    if not np.all(np.isreal(Whitesig)):
        raise Exception('Whitened matrix has imaginary values!') #should this be a warning
        
    #A - mixing matrix
    #W - demixing matrix
    if approach == 'Symm':
        AB, W = FastICASymm(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)
#         %timeit FastICASymm(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)
    else:
        A, W = FastICADefl(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)
    print('Here')
    
    return W@(mixedmean + m), AB, W

######################
from scipy.io.wavfile import read, write

rate, data = read('signals30.wav')
data = data.T.copy()
data = data.astype(np.float32)
A = np.load('A.npy')

frames = 0 #TODO
mixedsig = A[:,:]@data[:,:]

I, AB, W = FastICA(mixedsig, 'Symm', 50, 0.000001)