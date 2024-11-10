import numpy as np
from tqdm import tqdm
import copy
from utils import matrix_stack

# 化为X\beta进行adaptive lasso
def vector_adalasso(X,A,B,p,max_iter=5000,lam=1):
    def Xb_adala(X,A,B,p,max_iter=5000,lam=1):
        
        Phi = np.kron(B,A)
        temp = matrix_stack(X, p)
        Z_star = X[p:,:,:].reshape(X[p:,:,:].shape[0],-1,order='F').T
        X_star = temp[:-1,:,:].reshape(temp[:-1,:,:].shape[0],-1,order='F').T
        # Z_star = X[1:,:,:].reshape(X[1:,:,:].shape[0],-1,order='F').T
        # X_star = X[:-1,:,:].reshape(X[:-1,:,:].shape[0],-1,order='F').T
        y = Z_star.reshape(-1,1,order='F')
        I = np.identity(X.shape[1]*X.shape[2])
        X = np.kron(X_star.T,I)
        beta = Phi.reshape(-1,1,order='F')
        beta_hat = np.zeros(beta.shape)
        # beta_hat = beta
        
        tol = 1e-5
        for _ in tqdm(range(max_iter)):
            beta_old = copy.deepcopy(beta_hat)
            for k in range(beta_hat.shape[0]):
                rho = X[:,k].reshape(1,-1)@(y-X@beta_hat+X[:,k].reshape(-1,1)*beta_hat[k])
                z = (X[:,k]**2).sum()
                if beta[k] != 0 and z!=0:  # Avoid division by zero
                    beta_hat[k] = np.sign(rho) * max(abs(rho) - lam / abs(beta[k]), 0) / z
                else:
                    beta_hat[k] = 0
                # beta_hat[k] = np.sign(rho) * max(abs(rho)-lam/abs(beta[k]),0) / z
            if np.linalg.norm(beta_hat-beta_old, ord=1) < tol:
                break

        return beta_hat

    def index_01(X,p,beta_hat):
        
        def rearrange_kron_product(B_kron_A, A_shape, B_shape):
            """
            Convert B ⊗ A to A ⊗ B.
            """
            m, n = A_shape
            p, q = B_shape
            
            A_kron_B = np.zeros((m * p, n * q))
            
            for i in range(m):
                for j in range(n):
                    for k in range(p):
                        for l in range(q):
                            A_kron_B[i*p+k, j*q+l] = B_kron_A[i + k*m, j + l*n]
                            
            return A_kron_B
        
        m = X.shape[1]
        n = X.shape[2]
        
        A_hat = np.zeros([m,m*p])
        B_hat = np.zeros([n,n*p])
        B_A = beta_hat.reshape(m*n,m*n*p*p,order='F')
        
        for i in range(n):
            for j in range(n*p):
                if np.sum(B_A[i*m:i*m+m,j*m*p:j*m*p+m*p]) != 0:
                    B_hat[i,j] = 1
        
        A_B = rearrange_kron_product(B_A, A_hat.shape, B_hat.shape)
        for i in range(m):
            for j in range(m*p):
                if np.sum(A_B[i*n:i*n+n,j*n*p:j*n*p+n*p]) != 0:
                    A_hat[i,j] = 1
        
        # for i in range(m):
        #     for j in range(m):
        #         num = 0
        #         for k in range(i,m*n,m):
        #             for l in range(j,m*n,m):
        #                 # print(k,l)
        #                 num += B_A[k,l]
        #         if num != 0:
        #             A_hat[i,j] = 1
                    
        return A_hat,B_hat
    
    beta_hat = Xb_adala(X,A,B,p,max_iter,lam)
    A_hat,B_hat = index_01(X,p,beta_hat)

    return A_hat, B_hat

def vector_lasso(X,A,B,max_iter=5000,lam=1):
    def Xb_adala(X,A,B,max_iter=5000,lam=1):
        
        Phi = np.kron(B,A)
        Z_star = X[1:,:,:].reshape(X[1:,:,:].shape[0],-1,order='F').T
        X_star = X[:-1,:,:].reshape(X[:-1,:,:].shape[0],-1,order='F').T
        y = Z_star.reshape(-1,1,order='F')
        I = np.identity(X.shape[1]*X.shape[2])
        X = np.kron(X_star.T,I)
        beta = Phi.reshape(-1,1,order='F')
        beta_hat = np.zeros(beta.shape)
        # beta_hat = beta
        
        tol = 1e-5
        for _ in tqdm(range(max_iter)):
            beta_old = copy.deepcopy(beta_hat)
            for k in range(beta_hat.shape[0]):
                rho = X[:,k].reshape(1,-1)@(y-X@beta_hat+X[:,k].reshape(-1,1)*beta_hat[k])
                z = (X[:,k]**2).sum()
                beta_hat[k] = np.sign(rho) * max(abs(rho)-lam,0) / z #惩罚项直接更改lam就可以
            if np.linalg.norm(beta_hat-beta_old, ord=1) < tol:
                break

        return beta_hat

    def index_01(X,beta_hat):
        m = X.shape[1]
        n = X.shape[2]
        
        A_hat = np.zeros([m,m])
        B_hat = np.zeros([n,n])
        B_A = beta_hat.reshape(m*n,m*n,order='F')
        
        for i in range(n):
            for j in range(n):
                if np.sum(B_A[i*m:i*m+m,j*m:j*m+m]) != 0:
                    B_hat[i,j] = 1
                
        for i in range(m):
            for j in range(m):
                num = 0
                for k in range(i,m*n,m):
                    for l in range(j,m*n,m):
                        # print(k,l)
                        num += B_A[k,l]
                if num != 0:
                    A_hat[i,j] = 1
                    
        return A_hat,B_hat
    
    beta_hat = Xb_adala(X,A,B,max_iter,lam)
    A_hat,B_hat = index_01(X,beta_hat)

    return A_hat, B_hat

def vector_scad(X,A,B,a=3.7,lam=1,max_iter=5000):
    def Xb_adala(X,A,B,max_iter=5000,a=3.7,lam=1):
        
        Phi = np.kron(B,A)
        Z_star = X[1:,:,:].reshape(X[1:,:,:].shape[0],-1,order='F').T
        X_star = X[:-1,:,:].reshape(X[:-1,:,:].shape[0],-1,order='F').T
        y = Z_star.reshape(-1,1,order='F')
        I = np.identity(X.shape[1]*X.shape[2])
        X = np.kron(X_star.T,I)
        beta = Phi.reshape(-1,1,order='F')
        beta_hat = np.zeros(beta.shape)
        # beta_hat = beta
        
        tol = 1e-5
        for _ in tqdm(range(max_iter)):
            beta_old = copy.deepcopy(beta_hat)
            for k in range(beta_hat.shape[0]):
                rho = X[:,k].reshape(1,-1)@(y-X@beta_hat+X[:,k].reshape(-1,1)*beta_hat[k])
                z = (X[:,k]**2).sum()
                if abs(beta_hat[k]) <= lam:
                    beta_hat[k] = np.sign(rho) * max(abs(rho)-lam,0) / z
                elif abs(beta_hat[k]) > lam and abs(beta_hat[k]) <= a*lam:
                    beta_hat[k] = np.sign(rho) * max(abs(rho)-(a*lam-abs(beta_hat[k]))/(a-1),0) / z
                else:
                    beta_hat[k] = np.sign(rho) * max(abs(rho),0) / z
                
            if np.linalg.norm(beta_hat-beta_old, ord=1) < tol:
                break

        return beta_hat

    def index_01(X,beta_hat):
        m = X.shape[1]
        n = X.shape[2]
        
        A_hat = np.zeros([m,m])
        B_hat = np.zeros([n,n])
        B_A = beta_hat.reshape(m*n,m*n,order='F')
        
        for i in range(n):
            for j in range(n):
                if np.sum(B_A[i*m:i*m+m,j*m:j*m+m]) != 0:
                    B_hat[i,j] = 1
                
        for i in range(m):
            for j in range(m):
                num = 0
                for k in range(i,m*n,m):
                    for l in range(j,m*n,m):
                        # print(k,l)
                        num += B_A[k,l]
                if num != 0:
                    A_hat[i,j] = 1
                    
        return A_hat,B_hat
    
    beta_hat = Xb_adala(X,A,B,max_iter=max_iter,a=a,lam=lam)
    A_hat,B_hat = index_01(X,beta_hat)

    return A_hat, B_hat