import numpy as np
from tqdm import tqdm
import copy
from utils import matrix_stack


# 矩阵自回归的ols估计（坐标下降）
def final_mar_cd(X,p,A_hat,B_hat,A_ols=None,B_ols=None,max_iter=5000):
    """
    X:样本
    A_hat\B_hat 特征选择结果
    A_los\B_ols 坐标下降初始值
    max_iter 最大迭代次数
    """
    
    def m_ge_n1(A_hat,B_hat,X_1,X_2,max_iter):
        change = []
        tol = 1e-5
        index_a = np.where(A_hat==0)
        a_list = list(zip(index_a[0], index_a[1]))
        index_b = np.where(B_hat==0)
        b_list = list(zip(index_b[0], index_b[1]))
        
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(A_hat.shape[0]):
                if k < B_hat.T.shape[0]:
                    for l in range(A_hat.shape[1]):
                        if l < B_hat.T.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            if (k,l) not in a_list:
                                A_hat[k,l] = rho1 / z1
                            if (l,k) not in b_list: 
                                B_hat[l,k] = rho2 / z2
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            if (k,l) not in a_list:
                                A_hat[k,l] = rho1 / z1
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        if (k,l) not in a_list:
                            A_hat[k,l] = rho1 / z1
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n1(A_hat,B_hat,X_1,X_2,max_iter):
        change = []
        tol = 1e-5
        index_a = np.where(A_hat==0)
        a_list = list(zip(index_a[0], index_a[1]))
        index_b = np.where(B_hat==0)
        b_list = list(zip(index_b[0], index_b[1]))
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(B_hat.T.shape[0]):
                if k < A_hat.shape[0]:
                    for l in range(B_hat.T.shape[1]):
                        if l < A_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            if (k,l) not in a_list:
                                A_hat[k,l] = rho1 / z1
                            if (l,k) not in b_list: 
                                B_hat[l,k] = rho2 / z2
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            if (l,k) not in b_list: 
                                B_hat[l,k] = rho2 / z2
                else:
                    for l in range(B_hat.T.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        if (l,k) not in b_list: 
                            B_hat[l,k] = rho2 / z2
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change
    
    m = X.shape[1]
    n = X.shape[2]

    A_hat1 = copy.deepcopy(A_ols)
    B_hat1 = copy.deepcopy(B_ols)
    for i in range(A_hat.shape[0]):
        for j in range(A_hat.shape[1]):
            if A_hat[i,j] == 0:
                A_hat1[i,j] =0
    for i in range(B_hat.shape[0]):
        for j in range(B_hat.shape[1]):
            if B_hat[i,j] == 0:
                B_hat1[i,j] =0
    
    temp = matrix_stack(X, p)
    X_1 = temp[:-1,:,:]
    X_2 = X[p:,:,:]
    # X_1 = X[:-1,:,:]
    # X_2 = X[1:,:,:]
    
    if m >= n:
        A_final_ols, B_final_ols, change = m_ge_n1(A_hat1,B_hat1,X_1,X_2,max_iter)
    else:
        A_final_ols, B_final_ols, change = m_le_n1(A_hat1,B_hat1,X_1,X_2,max_iter)
    F_A1hat = np.linalg.norm(A_final_ols)
    A_final_ols = A_final_ols / F_A1hat
    B_final_ols = B_final_ols * F_A1hat
    
    return A_final_ols, B_final_ols



def final_mar_adalasso(X,A_01,B_01,A,B,p,lam1,lam2,max_iter=5000):
    
    """
    X 样本
    A,B adaptive lasso的自适应参数
    lam1,lam2 惩罚参数
    """
    
    def m_ge_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,max_iter):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(A_hat.shape[0]):
                if k < B_hat.T.shape[0]:
                    for l in range(A_hat.shape[1]):
                        if l < B_hat.T.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
            A_hat[A_01==0] = 0
            B_hat[B_01==0] = 0
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,max_iter):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(B_hat.T.shape[0]):
                if k < A_hat.shape[0]:
                    for l in range(B_hat.T.shape[1]):
                        if l < A_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                else:
                    for l in range(B_hat.T.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
            A_hat[A_01==0] = 0
            B_hat[B_01==0] = 0
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change
    
    m = X.shape[1]
    n = X.shape[2]
    A_hat = 0.1*np.ones(m*m*p).reshape(m,m*p)
    B_hat = 0.1*np.ones(n*n*p).reshape(n,n*p)
    # A_hat = np.zeros(m*m).reshape(m,m)
    # B_hat = np.zeros(n*n).reshape(n,n)
    # max_iter = 500
    temp = matrix_stack(X, p)
    X_1 = temp[:-1,:,:]
    X_2 = X[p:,:,:]
    
    if m >= n:
        A_hat, B_hat, change = m_ge_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,max_iter)
    else:
        A_hat, B_hat, change = m_le_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,max_iter)
    
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat / F_A1hat
    B_hat = B_hat * F_A1hat
    
    return A_hat, B_hat



def final_mar_adalasso2(X,A_01,B_01,A,B,p,lam1,lam2,delta,max_iter=5000):
    
    """
    X 样本
    A,B adaptive lasso的自适应参数
    lam1,lam2 惩罚参数
    """
    
    def m_ge_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,delta,max_iter):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(A_hat.shape[0]):
                if k < B_hat.T.shape[0]:
                    for l in range(A_hat.shape[1]):
                        if l < B_hat.T.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            if A_01[k,l] == 0:
                                A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/(abs(A[k,l])*delta),0) / z1
                            else:
                                A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
                            if B_01[l,k] == 0:
                                B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/(abs(B[l,k])*delta),0) / z2
                            else:
                                B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            if A_01[k,l] == 0:
                                A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/(abs(A[k,l])*delta),0) / z1
                            else:
                                A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        if A_01[k,l] == 0:
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/(abs(A[k,l])*delta),0) / z1
                        else:
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
            # A_hat[A_01==0] = 0
            # B_hat[B_01==0] = 0
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,delta,max_iter):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(B_hat.T.shape[0]):
                if k < A_hat.shape[0]:
                    for l in range(B_hat.T.shape[1]):
                        if l < A_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            if A_01[k,l] == 0:
                                A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/(abs(A[k,l])*delta),0) / z1
                            else:
                                A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / z1
                            if B_01[l,k] == 0:
                                B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/(abs(B[l,k])*delta),0) / z2
                            else:
                                B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            if B_01[l,k] == 0:
                                B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/(abs(B[l,k])*delta),0) / z2
                            else:
                                B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                else:
                    for l in range(B_hat.T.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        if B_01[l,k] == 0:
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/(abs(B[l,k])*delta),0) / z2
                        else:
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / z2
                            
            # A_hat[A_01==0] = 0
            # B_hat[B_01==0] = 0
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change
    
    m = X.shape[1]
    n = X.shape[2]
    A_hat = 0.1*np.ones(m*m*p).reshape(m,m*p)
    B_hat = 0.1*np.ones(n*n*p).reshape(n,n*p)
    # A_hat = np.zeros(m*m).reshape(m,m)
    # B_hat = np.zeros(n*n).reshape(n,n)
    # max_iter = 500
    temp = matrix_stack(X, p)
    X_1 = temp[:-1,:,:]
    X_2 = X[p:,:,:]
    
    if m >= n:
        A_hat, B_hat, change = m_ge_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,delta,max_iter)
    else:
        A_hat, B_hat, change = m_le_n_adala(A,B,A_hat,B_hat,A_01,B_01,X_1,X_2,lam1,lam2,delta,max_iter)
    
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat / F_A1hat
    B_hat = B_hat * F_A1hat
    
    return A_hat, B_hat