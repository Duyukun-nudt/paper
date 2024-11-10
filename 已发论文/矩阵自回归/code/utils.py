import numpy as np

def rate(A_true,B_true,A_hat,B_hat):
    """_summary_

    Args:
        A_true (_矩阵_): 真实的矩阵A
        B_true (_矩阵_): 真实的矩阵B
        A_hat (_矩阵_): 估计A
        B_hat (_矩阵_): 估计B
    """
    rate_A_mar_0 = np.sum((A_hat==(np.where(A_true==0,0,1)))&(A_true==0))/np.sum(A_true==0)
    rate_B_mar_0 = np.sum((B_hat==(np.where(B_true==0,0,1)))&(B_true==0))/np.sum(B_true==0)
    rate_A_mar_1 = np.sum((A_hat!=0)&(A_true!=0))/np.sum(A_true!=0)
    rate_B_mar_1 = np.sum((B_hat!=0)&(B_true!=0))/np.sum(B_true!=0)
    rate_mar_0 = (np.sum((A_hat==(np.where(A_true==0,0,1)))&(A_true==0))+np.sum((B_hat==(np.where(B_true==0,0,1)))&(B_true==0)))/(np.sum(A_true==0)+np.sum(B_true==0))
    rate_mar_1 = (np.sum((A_hat!=0)&(A_true!=0))+np.sum((B_hat!=0)&(B_true!=0)))/(np.sum(A_true!=0)+np.sum(B_true!=0))
    print('rate_A_mar_0: {}'.format(rate_A_mar_0))
    print('rate_B_mar_0: {}'.format(rate_B_mar_0))
    print('rate_A_mar_1: {}'.format(rate_A_mar_1))
    print('rate_B_mar_1: {}'.format(rate_B_mar_1))
    print('rate_mar_0: {}'.format(rate_mar_0))
    print('rate_mar_1: {}'.format(rate_mar_1))
    
def metrics_dif_kron(A_true,B_true,A_hat,B_hat):
    return np.log((np.linalg.norm(np.kron(B_hat,A_hat)-np.kron(B_true,A_true)))**2)

def matrix_stack(arr, t):
    k, m, n = arr.shape
    # 计算新的 k 维大小，即大矩阵的数量
    new_k = k - t + 1
    
    # 创建一个新的数组来存储最终结果
    result = np.zeros((new_k, m * t, n * t), dtype=arr.dtype)
    
    # 对于每个可能的起始索引 i
    for i in range(new_k):
        # 创建一个临时的大矩阵，用于存放合并的结果
        big_matrix = np.zeros((m * t, n * t), dtype=arr.dtype)
        for j in range(t):
            # 将 arr[i+j] 放在对应的对角线位置上
            row_start = j * m
            col_start = j * n
            big_matrix[row_start:row_start+m, col_start:col_start+n] = arr[i+j]
        
        # 将构建好的大矩阵存储到结果数组中
        result[i] = big_matrix
        
    return result

def genetate_data(dim_A=5,dim_B=6,T=10,p=1,p_1=0.5,uniform=[0.3,0.8],uniform_x=[1,10]):
    
    A_true = np.zeros(p*dim_A*dim_A)
    B_true = np.zeros(p*dim_B*dim_B)

    for i in range(0,p*dim_A*dim_A):
        A_true[i] = np.random.binomial(1, p_1, size=1)*np.random.uniform(uniform[0],uniform[1])
    for i in range(0,p*dim_B*dim_B):
        B_true[i] = np.random.binomial(1, p_1, size=1)*np.random.uniform(uniform[0],uniform[1])
            
    A_true = A_true.reshape(p,dim_A,dim_A)
    B_true = B_true.reshape(p,dim_B,dim_B)
    for i in range(p):
        F_A1hat = np.linalg.norm(A_true[i,:,:])
        A_true[i,:,:] = A_true[i,:,:]/F_A1hat
        B_true[i,:,:] = B_true[i,:,:]*F_A1hat

    X = np.random.uniform(uniform_x[0],uniform_x[1],[T,dim_A,dim_B])
    for i in range(p,T):
        # X[i] = np.sum(A_true@(X[i-t:i,:,:])@B_true,axis=0)+np.random.normal(0,0.3,(X.shape[1],X.shape[2]))
        X[i] = np.sum(A_true@(X[i-p:i,:,:])@B_true.transpose(0,2,1),axis=0)+np.random.randn(X.shape[1],X.shape[2])
        
    A = A_true[0,:,:]
    for i in range(1,p):
        A = np.hstack((A,A_true[i,:,:]))
    
    B = B_true[0,:,:]
    for i in range(1,p):
        B = np.hstack((B,B_true[i,:,:]))

    return A,B,X
