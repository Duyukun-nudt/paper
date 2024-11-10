import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import math
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

from scipy.optimize import linear_sum_assignment

from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import KDTree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp

def efficient_min_distance_method(samples,p):
    tree = KDTree(samples)
    min_distance = np.inf
    closest_pair = (0, 0)

    for i in range(samples.shape[0]):
        distances, indices = tree.query(samples[i], k=2, p=p)  # k=2 返回最近的两个点（包括自身）
        if distances[1] < min_distance:  # distances[1] 是第二近的点（即除了自身外最近的点）
            min_distance = distances[1]
            closest_pair = (i, indices[1])

    return closest_pair

def K_Space(cluster_nums, samples, p=2):

    n = samples.shape[0]
    num_each_cluster = n // cluster_nums
    clusters = [[] for _ in range(cluster_nums+1)]
    cluster_centers = []
    samples_remaining = samples.copy()
    add_cluster_index = 0
    clusters_ = []
    time = 0

    while samples_remaining.shape[0] + len(cluster_centers)>1:
        #打印每次迭代的效果
        
        # print(f"第{time}次循环 samples_reminding:{samples_remaining.shape[0]} 临时类数量:{len(cluster_centers)} 产出类数量:{len(clusters_)}")
        
        matrix = np.vstack([samples_remaining,[i for i in cluster_centers]]) if len(cluster_centers)>0 else samples_remaining
        #首先计算合并的矩阵中距离最近的两个元素
        index1, index2 = efficient_min_distance_method(matrix,p=p)
        #求得两个元素是样本还是簇心
        label1 = 0 if index1 < samples_remaining.shape[0] else 1 # 0表示样本,1表示簇心
        label2 = 0 if index2 < samples_remaining.shape[0] else 1
        
        #样本点之间的合并
        if add_cluster_index < len(clusters) and label1+label2 == 0:
            clusters[add_cluster_index].extend([samples_remaining[index1], samples_remaining[index2]])
            cluster_centers.append(np.mean(clusters[add_cluster_index], axis=0))
            samples_remaining = np.delete(samples_remaining, [index1, index2], axis=0)
            add_cluster_index += 1
            # print(f"样本合并创建新类 ")
            
        #当簇的数量足够时,如果还是两个样本,则将两个样本作为一个整体加入距离簇心最近的类中
        elif add_cluster_index == len(clusters) and label1+label2 == 0:
            two_samples_mean = np.mean([samples_remaining[index1], samples_remaining[index2]],axis=0)
            # print(cluster_centers)
            # print(clusters)
            # print(clusters_)
            if add_cluster_index == 0 and len(clusters) == 0:
                clusters.append([])
                clusters.append([])
                clusters[add_cluster_index].extend([samples_remaining[index1], samples_remaining[index2]])
                cluster_centers.append(np.mean(clusters[add_cluster_index], axis=0))
                samples_remaining = np.delete(samples_remaining, [index1, index2], axis=0)
                add_cluster_index += 1
                # print(f"类数已充足,增加类数")
            else:
                cluster_centers_matrix = np.vstack([i for i in cluster_centers])
                tree = KDTree(cluster_centers_matrix)
                _, indices = tree.query(two_samples_mean, k=1, p=p)
                clusters[indices].extend([samples_remaining[index1], samples_remaining[index2]])
                cluster_centers[indices] = np.mean(clusters[indices], axis=0)
                samples_remaining = np.delete(samples_remaining, [index1,index2], axis=0)
                # print(f"两个样本归纳新类")
            
        #样本点添加到类中
        elif label1+label2 == 1:
            index_samples = index1 if label1 == 0 else index2
            index_cluster = index1-samples_remaining.shape[0] if label1 == 1 else index2-samples_remaining.shape[0]
            clusters[index_cluster].append(samples_remaining[index_samples])
            cluster_centers[index_cluster] = np.mean(clusters[index_cluster], axis=0)
            samples_remaining = np.delete(samples_remaining, index_samples, axis=0)
            # print(f"一个样本归纳新类")
           
        #类与类之间的合并
        else:
            index1 = index1 - samples_remaining.shape[0]
            index2 = index2 - samples_remaining.shape[0]
            if len(clusters[index1])+len(clusters[index2]) <= num_each_cluster:#如果两个类中样本合并的数量不足饱和
                clusters[index1] = clusters[index1]+clusters[index2]
                cluster_centers[index1] = np.mean(clusters[index1], axis=0)
                del clusters[index2]
                del cluster_centers[index2]
                add_cluster_index -= 1
            #使用lof算法剔除多余样本点
            else:
                points = np.vstack([clusters[index1]+clusters[index2]])
                del clusters[max(index1,index2)]
                del clusters[min(index1,index2)]
                del cluster_centers[max(index1,index2)]
                del cluster_centers[min(index1,index2)]
                lof_model = LocalOutlierFactor(n_neighbors=int(num_each_cluster/2),p=p)
                lof_model.fit_predict(points)
                score = abs(lof_model.negative_outlier_factor_)
                clusters_.append(list(points[np.argsort(score)[:num_each_cluster]]))
                samples_remaining = np.vstack([samples_remaining,points[np.argsort(score)[num_each_cluster:]]])
                add_cluster_index -= 2
            clusters.append([])
            # print(f"两个类合并")
            
        time += 1
        #剔除已经饱和的簇
        for i in range(add_cluster_index):
            if len(clusters[i]) >= num_each_cluster:
                clusters_.append(clusters[i])
                del clusters[i]
                del cluster_centers[i]
                add_cluster_index -= 1
                break
        #将最后剩余的样本点归纳为一个类
        if samples_remaining.shape[0] + len(cluster_centers) == 1:
            if len(clusters) == 0:
                break
            clusters_.append(clusters[0])
            

    return clusters_

#计算聚类效果
def calculate_indicator(samples,label):
    clusters_center = []
    for i in range(np.max(label+1)):
        center = np.mean(samples[np.where(label==i)],axis=0)
        clusters_center.append(center)
    clusters_center = np.array(clusters_center)
    m = 0
    for i in range(samples.shape[0]):
        distance = np.linalg.norm(samples[i,:]-clusters_center,axis=1)
        if np.argmin(distance) != label[i]:
            m += 1
            
    return m/samples.shape[0]
    
def list_transform_array(clusters):
    x_train = []
    y_train = []
    max_depth = int(len(clusters)/3)
    #将原本聚类结果完全转化为列表,导出label
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            x_train.append(clusters[i][j])
            y_train.append(i)
    #将列表格式的数据转化为数组
    x_train = np.vstack(x_train)
    y_train = np.array(y_train)
    return x_train,y_train
    
def fine_turning(x_train,y_train):
    n,m = x_train.shape
    nums_class = np.max(y_train)+1
    #计算优化前的指标
    indicator = calculate_indicator(x_train,y_train)
    # print(f"优化前聚类效果指标:{indicator}")
    #选择机器学习模型进行预测,调整聚类结果
    knn = KNeighborsClassifier(n_neighbors=int(n/nums_class))
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_train)
    # print(f"优化样本占比:{round((1-accuracy_score(y_train, predictions))*100,5)}%\n优化后聚类效果指标:{calculate_indicator(x_train,predictions)}")
    
    c = predictions
    while next(x for x in range(len(c)+1) if x not in c) <= c.max():
        missing_number = next(x for x in range(len(c)) if x not in c)
        c = np.array([x - 1 if x > missing_number else x for x in c])
    
    return x_train,c

def caculater_cluster_index(samples,cluster_label):
    cluster_center = []
    for i in range(max(cluster_label)+1):
        cluster = samples[np.where(cluster_label == i)]
        cluster_center.append(np.mean(cluster,axis=0))
    distance_matrix_ = distance_matrix(cluster_center,cluster_center)
    path = solve_tsp( distance_matrix_, endpoints = (0,0) )
    return path

def transform(b,c,index):
    conclusion = []
    for i in range(max(index)+1):
        number = index[i]
        conclusion.append(np.vstack(b[np.where(c==number)]))
    
    return conclusion

def match1(data, regular=True):
    # 输入的data需要时簇心排序好的list
    ''' 
        input:  list data 里面是每一类的array（簇心已排序好）
                torch: 是否考虑上一条线 默认为True
        output: list Para 里面是相邻两类的拟合直线参数
    '''
    if not regular:
        Para = []
        for i in range(len(data)-1):
            model = LinearRegression()
            d = np.concatenate([data[i],data[i+1]])
            model.fit(d[:,:-1], d[:,-1])
            para = np.append(model.coef_,model.intercept_)
            Para.append(para)
    
    else:
        # 拟合第一条线
        Para = []
        model = LinearRegression()
        d = np.concatenate([data[0],data[1]])
        # model.fit(data[0][:,:-1], data[0][:,-1])
        model.fit(d[:,:-1], d[:,-1])
        para = np.append(model.coef_,model.intercept_)
        Para.append(para)
        # 簇心
        center = []
        for i in range(len(data)):
            center.append(np.mean(data[i][:,:-1],axis=0))

        for i in tqdm(range(1,len(data)-1)):
            p = torch.randn(len(Para[0]), requires_grad=True)
            p_ = torch.tensor(Para[i-1]).float()
            d = torch.tensor(np.concatenate([data[i],data[i+1]])).float()
            epoch_num = 3000
            optimizer = torch.optim.Adam([p],lr=0.01)
            # lam = 0.5 # 惩罚项系数
            temp_min = ((np.array(center)[1:]-np.array(center)[:-1])**2).min()
            temp_max = ((np.array(center)[1:]-np.array(center)[:-1])**2).max()
            lam = (temp_max - (center[i]-center[i-1])**2) / (temp_max-temp_min) * 0.5
            lam = lam[0]

            for epoch in range(epoch_num):
                optimizer.zero_grad()
                L = ((torch.matmul(d[:,:-1],p[:-1].reshape(-1,1))+p[-1]-d[:,-1].reshape(-1,1))**2).mean() + lam*((p-p_)**2).mean()
                L.backward()
                optimizer.step()

            Para.append(p.detach().numpy())
    return Para

def match2(A,B):
    ''' 
        input:  list A 第一类每个点的误差
                list B 第二类每个点的误差
        output: array row_ind 第一类的匹配点索引
                array col_ind 第二类的匹配点索引
                total_cost  总成本
    '''
    # 提供的成本矩阵
    cost_matrix = np.array([[abs(a + b) for b in B] for a in A])

    # 获取原始矩阵的行数和列数
    rows, cols = cost_matrix.shape

    # 扩展成本矩阵为方阵
    if rows == cols:
        extended_matrix = cost_matrix
    else:
        size = max(rows, cols)
        extended_matrix = np.full((size, size), np.max(cost_matrix) + 1) # 使用最大值+1来扩充成本矩阵
        extended_matrix[:rows, :cols] = cost_matrix

    # 应用匈牙利算法
    row_ind, col_ind = linear_sum_assignment(extended_matrix)

    # 初始化总成本
    total_cost = 0

    # 仅使用原始矩阵范围内的匹配来计算总成本
    total_cost = extended_matrix[row_ind, col_ind].sum()
    total_cost = total_cost - (np.max(cost_matrix) + 1)*np.abs(len(B)-len(A))
    if len(A) > len(B):
        for j in range(np.abs(len(B)-len(A))):
            index = row_ind[np.where(col_ind==len(A)-1-j)[0][0]]
            col_ind[index] = np.argmin(cost_matrix[index,:])
            total_cost += np.min(cost_matrix[index,:])
    elif len(A) < len(B):
        for j in range(np.abs(len(B)-len(A))):
            index = row_ind[np.where(row_ind==len(B)-1-j)[0][0]]
            row_ind[index] = np.argmin(cost_matrix[:,col_ind[index]])
            total_cost += np.min(cost_matrix[:,col_ind[index]])

    # 输出结果
    return row_ind, col_ind, total_cost

def interpolation(samples, y, k, eta, regular=True):
    a = K_Space(k,samples)
    b,c = list_transform_array(a)
    b,c = fine_turning(b,c)
    index = caculater_cluster_index(b,c)
    conclusion = transform(b,c,index)
    
    for i in range(len(conclusion)):
        indices = np.array([np.where((samples == row).all(axis=1))[0][0] for row in conclusion[i]])
        conclusion[i] = np.concatenate([conclusion[i], y[indices]],axis=1)

    data = conclusion
    Para = match1(data, regular)
    Row_ind, Col_ind, Total_cost = [], [], []
    for i in range(len(Para)):
        A = list((np.matmul(data[i][:,:-1], Para[i][:-1].reshape(-1,1)) + Para[i][-1] - data[i][:,-1].reshape(-1,1)).reshape(-1))
        B = list((np.matmul(data[i+1][:,:-1], Para[i][:-1].reshape(-1,1)) + Para[i][-1] - data[i+1][:,-1].reshape(-1,1)).reshape(-1))
        row_ind, col_ind, total_cost = match2(A,B)
        Row_ind.append(row_ind)
        Col_ind.append(col_ind)
        Total_cost.append(total_cost)

    samples_add = np.concatenate([samples,y],axis=1)
    distance_sum = 0
    for i in range(len(Row_ind)):
        for j in range(len(Row_ind[i])):
            point1 = data[i][Row_ind[i][j]]
            point2 = data[i+1][Col_ind[i][j]]
            distance_sum += np.linalg.norm(point2-point1,ord=2)
    
    n = samples.shape[0]
    for i in range(len(Row_ind)):
        if (i != 8)&(i!=20):
            for j in range(len(Row_ind[i])):
                point1 = data[i][Row_ind[i][j]]
                point2 = data[i+1][Col_ind[i][j]]
                distance = np.linalg.norm(point2-point1,ord=2)
                num_samples = int(np.round(eta*n*distance/distance_sum))
                sample_points = np.linspace(point1, point2, num_samples+ 2)[1:-1]
                samples_add = np.concatenate([samples_add,sample_points])
    
    return data, Para, samples_add