import numpy as np
from scipy.sparse.linalg import svds,eigs
from scipy.linalg import lstsq
from scipy.linalg import norm
from scipy import sparse
from numpy import matlib
import math
eps_l = np.finfo(float).eps

def genSubspace(D, n, Ni, di, sigma=0, corruption=0.01):
    '''
    This creates a matrix X with shape D by sum(Ni) that contains n subspaces
    of dimension di, with noise level sigma.
    :param args:
        D: dimension of ambience space
        n: number of subspace
        Ni: points in each subspace
        di: dimension of each subspace
        sigma: noise deviation
        corruption: error deviation
    :return:
        X: data (D by sum(Ni))
        s: label of X (1 by sum(Ni))
    '''

    '''if len(args)<6:
        corruption = 0
    if len(args)<5:
        sigma = 0
    if len(args)==6:
        [D, n, Ni, di, sigma, corruption] = [args[0],args[1],args[2],args[3],args[4],args[5]]
    else:
        [D, n, Ni, di] = [args[0],args[1],args[2],args[3]]
    '''
    di = di*np.ones((1,n), dtype = np.int64)


    X = np.zeros((D, int(np.sum(Ni))))
    s = np.zeros((1, int(np.sum(Ni))))
    idx = 0

    for j in range(n):
        Xtmp = np.random.randn(D, D)
        u,_,_ = svds(Xtmp, di[0,j])
        #print(di.dtype, Ni.dtype)
        v = np.random.randn(di[0,j],Ni[0,j])
        Xtmp = np.matmul(u,v)
        #print(Xtmp)
        y = np.sqrt(np.sum(Xtmp**2,0))
        Xtmp = np.divide(Xtmp,y)
        #print(type(Ni[0,j]))
        X[:, idx:idx+Ni[0,j]] = Xtmp
        s[0,idx:idx+Ni[0,j]] = j
        idx = idx + Ni[0,j]

    noise_term = sigma*np.random.randn(D, np.sum(Ni))/np.sqrt(D)
    X = X+noise_term
    corruption_mask_inter = np.random.permutation(D*int(np.sum(Ni)))
    #print(corruption_mask_inter[0])
    #print(int(round(corruption*D*int(np.sum(Ni)),0)))
    corruption_mask = corruption_mask_inter[0:int(round(corruption*D*int(np.sum(Ni)),0))]
    #print(X.shape)
    #print(corruption_mask)
    X.flat[corruption_mask] = 0
    #print(X.shape)
    #print(s.shape)
    return X,s

def Ompmat(X,K,thr):            #not working
    '''

    :param X:
    :param K:
    :param thr:
    :return:
    '''
    MEMORY_TOTAL = 0.1*(10**9)
    _, N = X.shape

    Xn = X
    Xn = cnormalize(X)
    S = np.ones((N,K), dtype = np.int64)
    int_rge = np.arange(N)
    Ind = np.matlib.repmat(int_rge, K, 1)
    Ind = Ind.transpose()
    Val = np.zeros((N,K))
    t_vec = K*np.ones((N,1), dtype=np.int64)
    res = np.copy(Xn)
    for t in range(K):
        blockSize = round(MEMORY_TOTAL / N)
        counter = 0
        while(1):
            mask = range(counter,min(counter + blockSize, N))
            I = abs(np.matmul((X.transpose()),res[:,mask]))
            np.fill_diagonal(I, 0)
            J = np.argmax(I, axis=0)
            S[mask,t] = J
            counter = counter+blockSize
            if counter >= N:
                break

        print(S)
        if t+1 != K:
            for iN in range(N):
                if t_vec[iN] == K:
                    B = Xn[:,S[iN,0:(t+1)]]    #check here
                    sol,_,_,_ = lstsq(B,Xn[:,iN])
                    res[:,iN] = Xn[:,iN] - np.matmul(B,sol)     #problem
                    if np.sum(res[:,iN]**2)<thr:
                        t_vec[iN] = t
        print(S)
        if np.any(t_vec==K) == False:
            print(t_vec)
            break

    for iN in range(N):
        #print(t_vec)
        inter,_,_,_ =  lstsq(X[:,S[iN,0:np.asscalar(t_vec[iN]+1)]],X[:,iN])
        Val[iN,0:np.asscalar(t_vec[iN])] = inter.transpose()

    C = sparse.coo_matrix((Val.flat,(S.flat,Ind.flat)),shape=(N,N))
    return C

def cnormalize(X,p=2):
    '''
    This function normalized the columns of the given
    matrix.
    :param X: The given matrix X
    :param p: This gives the norm
    :return:
    Y:returns the data in Y
    Xnorm: returns the norm values in Xnorm
    '''

    Xnorm = X / (np.linalg.norm(X, ord = p,axis = 0)+eps_l)
    return Xnorm

def evalSSR_perc(C,s):
    N = s.shape[0]
    x = np.zeros(N)
    for ii in range(N):
        x[ii] = np.max(np.abs(C[s!=s[ii], ii]))
    x = x/np.max(np.abs(C),axis=0)
    #print(x)
    perc = np.sum(x<1e-5)/N
    return perc,x

def evalSSR_error(C,s):
    N = s.shape[0]

    error = 0
    #e_vec = np.zeros((1,N))
    e_vec = np.zeros(N)
    for iN in range(N):
        if norm(C[:,iN],ord=1)<eps_l:
            error = error+1
        else:
            error = error+norm(C[s!=s[iN], iN], ord = 1)/norm(C[:,iN], ord = 1)
        e_vec[iN] = norm(C[s!=s[iN], iN], ord = 1)/norm(C[:,iN], ord = 1)
    error = error/N
    return error

def evalConn(C,s):
    #isSymmetric warning not implemented
    s_val = np.unique(s)
    n = s_val.shape[0]

    conn = np.Inf
    for iN in range(n):
        C_in = C[s == s_val[iN],:][:, s == s_val[iN]]
        if np.min(np.sum(C_in, axis=1))<eps_l:  #check this
            conn_in = 0.0
        else:
            B = cnormalize(C_in, 1).transpose()
            eig_in, _ = eigs(A = B, k=2, which='LR', tol=1e-3)
            conn_in = 1-np.abs(eig_in[1])
        conn = np.minimum(conn,conn_in)
    return conn

def evalAccuracy(label1, label2):
    label2 = bestMap(label1, label2);
    accuracy = np.sum(label1.flat == label2.flat) / label2.shape[0]
    return accuracy