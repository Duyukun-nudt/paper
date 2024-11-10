import numpy as np
import copy
from tqdm import tqdm
from sklearn import linear_model 
from utils import matrix_stack

def mar_cd(X,p=1,max_iter = 5000):
    def m_ge_n(A_hat,B_hat,X_1,X_2,max_iter):
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
                            A_hat[k,l] = rho1 / z1
                            B_hat[l,k] = rho2 / z2
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            A_hat[k,l] = rho1 / z1
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        A_hat[k,l] = rho1 / z1
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n(A_hat,B_hat,X_1,X_2,max_iter):
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
                            A_hat[k,l] = rho1 / z1
                            B_hat[l,k] = rho2 / z2
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            B_hat[l,k] = rho2 / z2
                else:
                    for l in range(B_hat.T.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        B_hat[l,k] = rho2 / z2
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change
    m = X.shape[1]
    n = X.shape[2]
    A_hat = 0.1*np.ones(m*m*p).reshape(m,m*p)
    B_hat = 0.1*np.ones(n*n*p).reshape(n,n*p)

    # X_1 = X[:-1,:,:]
    Z = matrix_stack(X,p)
    X_1 = Z[:-1,:,:]
    X_2 = X[p:,:,:]
    
    if m >= n:
        A_hat, B_hat, change = m_ge_n(A_hat,B_hat,X_1,X_2,max_iter)
    else:
        A_hat, B_hat, change = m_le_n(A_hat,B_hat,X_1,X_2,max_iter)
    
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat/F_A1hat
    B_hat = B_hat*F_A1hat
    
    return A_hat, B_hat



def mar_PROJ(X,p=1):
    # 重排列算子
    def G(Phi, shape_A, shape_B):
        # 获取矩阵的维度
        m, n = shape_A
        d, q = shape_B

        # 检查Kronecker积矩阵的形状是否正确
        if Phi.shape != (m * d, n * q):
            raise ValueError("Kronecker积结果矩阵Phi的形状与给定的形状不匹配")

        # 初始化Phi_tilde矩阵
        Phi_tilde = np.zeros((m * n, d * q))

        count = 0
        for i in range(q):
            for j in range(d):
                Phi_tilde[:, count] = Phi[j * m:(j + 1) * m, i * n:(i + 1) * n].reshape(m * n, order='F')
                count += 1

        return Phi_tilde
    _,m,n = X.shape # 获取数据维度
    
    temp = matrix_stack(X, p)
    Z_star = X[p:,:,:].reshape(X[p:,:,:].shape[0],-1,order='F').T
    X_star = temp[:-1,:,:].reshape(temp[:-1,:,:].shape[0],-1,order='F').T
    
    # Z_star = X[1:,:,:].reshape(X[1:,:,:].shape[0],-1,order='F').T
    # X_star = X[:-1,:,:].reshape(X[:-1,:,:].shape[0],-1,order='F').T
    y = Z_star.reshape(-1,1,order='F')
    # I = np.identity(temp.shape[1]*temp.shape[2])

    I = np.identity(X.shape[1]*X.shape[2])
    X = np.kron(X_star.T,I)
    model = linear_model.LinearRegression(fit_intercept=False)
    
    model.fit(X,y)
    beta = model.coef_
    Phi = beta.reshape(m*n,m*n*p*p,order='F')
    Phi_tilde = G(Phi,(m,m*p),(n,n*p))
    # print(Phi.shape,Phi_tilde.shape)
    # Phi = beta.reshape(m*n,m*n,order='F')
    # Phi_tilde = G(Phi,m,n)
    U1, Sigma1, V1 = np.linalg.svd(Phi_tilde)#奇异值分解
    
    A1hat = U1[:,0].reshape(m,m*p,order='f')
    B1hat = Sigma1[0] * V1[0,:].reshape(n,n*p,order='f')
    F_A1hat = np.linalg.norm(A1hat)
    A1hat = A1hat / F_A1hat
    B1hat = B1hat * F_A1hat
    # print(Sigma1[0])

    return A1hat,B1hat

def mar_iter_ols(Xt,p,max_iter=5000):
    temp = matrix_stack(Xt, p)
    x = temp[:-1,:,:]
    y = Xt[p:,:,:]
    # x = Xt[:-1,:,:]
    # y = Xt[1:,:,:]
    A_proj,B_proj = mar_PROJ(Xt,p)
    A_hat,B_hat = copy.deepcopy(A_proj), copy.deepcopy(B_proj)
    tol = 1e-5
    for _ in tqdm(range(max_iter)):
        B_old = copy.deepcopy(B_hat)
        A_old = copy.deepcopy(A_hat)
        B_hat = np.sum(y.transpose(0,2,1)@A_hat@x,axis=0)@np.linalg.inv(np.sum(x.transpose(0,2,1)@(A_hat.T)@A_hat@x,axis=0))
        A_hat = np.sum(y@B_hat@x.transpose(0,2,1),axis=0)@np.linalg.inv(np.sum(x@(B_hat.T)@B_hat@x.transpose(0,2,1),axis=0))
        dif = np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1)
        if  dif< tol :
            break   
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat / F_A1hat
    B_hat = B_hat * F_A1hat
    return A_hat,B_hat
    


# 矩阵自回归adaptive lasso（坐标下降）
# 矩阵自回归adaptive lasso（坐标下降）
def mar_adalasso(X,A,B,p,lam1,lam2,max_iter):
    
    """
    X 样本
    A,B adaptive lasso的自适应参数
    lam1,lam2 惩罚参数
    """
    
    def m_ge_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter):
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
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / (z1+1e-5)
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / (z2+1e-5)
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / (z1+1e-5)
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / (z1+1e-5)
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter):
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
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1/abs(A[k,l]),0) / (z1+1e-5)
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / (z2+1e-5)
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / (z2+1e-5)
                else:
                    for l in range(B_hat.T.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2/abs(B[l,k]),0) / (z2+1e-5)
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
        A_hat, B_hat, change = m_ge_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter)
    else:
        A_hat, B_hat, change = m_le_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter)
    
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat / F_A1hat
    B_hat = B_hat * F_A1hat
    
    return A_hat, B_hat


def mar_lasso(X,A,B,lam1,lam2):
    
    """
    X 样本
    A,B adaptive lasso的自适应参数
    lam1,lam2 惩罚参数
    """
    
    def m_ge_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(A_hat.shape[0]):
                if k < B_hat.shape[0]:
                    for l in range(A_hat.shape[1]):
                        if l < B_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1,0) / z1
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2,0) / z2
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1,0) / z1
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1,0) / z1
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(B_hat.shape[0]):
                if k < A_hat.shape[0]:
                    for l in range(B_hat.shape[1]):
                        if l < A_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            A_hat[k,l] = np.sign(rho1) * max(abs(rho1)-lam1,0) / z1
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2,0) / z2
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2,0) / z2
                else:
                    for l in range(B_hat.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        B_hat[l,k] = np.sign(rho2) * max(abs(rho2)-lam2,0) / z2
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change
    
    m = X.shape[1]
    n = X.shape[2]
    A_hat = 0.1*np.ones(m*m).reshape(m,m)
    B_hat = 0.1*np.ones(n*n).reshape(n,n)
    # A_hat = np.zeros(m*m).reshape(m,m)
    # B_hat = np.zeros(n*n).reshape(n,n)
    max_iter = 5000
    X_1 = X[:-1,:,:]
    X_2 = X[1:,:,:]
    
    if m >= n:
        A_hat, B_hat, change = m_ge_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter)
    else:
        A_hat, B_hat, change = m_le_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter)
    
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat / F_A1hat
    B_hat = B_hat * F_A1hat
    
    return A_hat, B_hat

def mar_scad(X,A,B,lam1,lam2,a=3.7,max_iter = 5000):
    
    """
    X 样本
    A,B adaptive lasso的自适应参数
    lam1,lam2 惩罚参数
    a scad超参数
    """
    def scad(beta1,rho,z,lam=1,a=3.7):
        beta = copy.deepcopy(beta1)
        if abs(beta) <= lam:
            beta = np.sign(rho) * max(abs(rho)-lam,0) / z
        elif abs(beta) > lam and abs(beta) <= a*lam:
            beta = np.sign(rho) * max(abs(rho)-(a*lam-abs(beta))/(a-1),0) / z
        else:
            beta = np.sign(rho) * max(abs(rho),0) / z
        return beta
    
    def m_ge_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter,a):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(A_hat.shape[0]):
                if k < B_hat.shape[0]:
                    for l in range(A_hat.shape[1]):
                        if l < B_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            A_hat[k,l] = scad(A_hat[k,l],rho1,z1,lam1,a)
                            B_hat[l,k] = scad(B_hat[l,k],rho2,z2,lam2,a)
                        else:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            A_hat[k,l] = scad(A_hat[k,l],rho1,z1,lam1,a)
                else:
                    for l in range(A_hat.shape[1]):
                        rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                        z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                        A_hat[k,l] = scad(A_hat[k,l],rho1,z1,lam1,a)
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change

    def m_le_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter,a):
        change = []
        tol = 1e-5
        for iter in tqdm(range(max_iter)):
            A_old = copy.deepcopy(A_hat)
            B_old = copy.deepcopy(B_hat)
            for k in range(B_hat.shape[0]):
                if k < A_hat.shape[0]:
                    for l in range(B_hat.shape[1]):
                        if l < A_hat.shape[1]:
                            rho1 = np.sum((X_1@B_hat.T)@X_2.transpose(0,2,1),axis=0)[l,k] - np.sum(((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1))@A_hat.T,axis=0)[l,k] + A_hat[k,l]*np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l]
                            z1 = np.sum((X_1@(B_hat.T@B_hat))@X_1.transpose(0,2,1),axis=0)[l,l] 
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            A_hat[k,l] = scad(A_hat[k,l],rho1,z1,lam1,a)
                            B_hat[l,k] = scad(B_hat[l,k],rho2,z2,lam2,a)
                        else:
                            rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                            B_hat[l,k] = scad(B_hat[l,k],rho2,z2,lam2,a)
                else:
                    for l in range(B_hat.shape[1]):
                        rho2 = np.sum((X_1.transpose(0,2,1)@A_hat.T)@X_2,axis=0)[k,l] - np.sum(((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1)@B_hat.T,axis=0)[k,l] + (B_hat.T)[k,l]*np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        z2 = np.sum((X_1.transpose(0,2,1)@(A_hat.T@A_hat))@X_1,axis=0)[k,k]
                        B_hat[l,k] = scad(B_hat[l,k],rho2,z2,lam2,a)
            change.append(np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1))
            if np.linalg.norm(A_hat-A_old,ord=1)+np.linalg.norm(B_hat-B_old,ord=1) < tol:
                break    
        return A_hat, B_hat, change
    
    m = X.shape[1]
    n = X.shape[2]
    A_hat = 0.1*np.ones(m*m).reshape(m,m)
    B_hat = 0.1*np.ones(n*n).reshape(n,n)
    # A_hat = np.zeros(m*m).reshape(m,m)
    # B_hat = np.zeros(n*n).reshape(n,n)
    X_1 = X[:-1,:,:]
    X_2 = X[1:,:,:]
    
    if m >= n:
        A_hat, B_hat, change = m_ge_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter,a)
    else:
        A_hat, B_hat, change = m_le_n_adala(A,B,A_hat,B_hat,X_1,X_2,lam1,lam2,max_iter,a)
    
    F_A1hat = np.linalg.norm(A_hat)
    A_hat = A_hat / F_A1hat
    B_hat = B_hat * F_A1hat
    
    return A_hat, B_hat