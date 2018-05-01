# -*- coding: utf-8 -*-
# @Author  : Peidong
# @Site    :
# @File    : recommendsystem.py
# @Software: PyCharm
# 使用MovieLens数据集，它是在实现和测试推荐引擎时所使用的最常见的数据集之一。它包含来自于943个用户
# 以及精选的1682部电影的100K个电影打分。
import numpy as np
import pandas as pd

# 读取u.data文件
header = ['user_id', 'item_id', 'rating', 'timestamp']
# df = pd.read_csv('./DataSets/MovieLens/ml-20m/ratings.csv', sep=',', names=header)
df = pd.read_csv('./ratings.csv', sep=',', names=header)
df.drop(0, axis=0, inplace=True)
# print (df)
n_users = df.user_id.unique().shape[0]
# 计算唯一用户和电影的数量
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

UserList = []
UserMap = {}
UserList = list(set(df.user_id))
i = 0
for value in UserList:
    UserMap[value] = i
    i += 1
MovieList = []
MovieMap = {}
i = 0
MovieList = list(set(df.item_id))
for value in MovieList:
    MovieMap[value] = i
    i += 1

# 使用scikit-learn库将数据集分割成测试和训练。Cross_validation.train_test_split根据测试样本的比例（test_size）
# ，本例中是0.25，来将数据混洗并分割成两个数据集
from sklearn import model_selection as ms

train_data, test_data = ms.train_test_split(df, test_size=0.2)
# print ("train_data: " + str(train_data))

# 基于内存的协同过滤
# 第一步是创建uesr-item矩阵，此处需创建训练和测试两个UI矩阵
train_data_matrix = np.zeros((n_users, n_items))

for line in train_data.itertuples():
    i = UserMap[line[1]]
    j = MovieMap[line[2]]
    train_data_matrix[i, j] = line[3]
print ("train_data_matrix" + str(train_data_matrix.shape))

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    i = UserMap[line[1]]
    j = MovieMap[line[2]]
    test_data_matrix[i, j] = line[3]
print ("test_data_matrix" + str(test_data_matrix.shape))
# print(test_data_matrix)

# 计算相似度
# 使用sklearn的pairwise_distances函数来计算余弦相似性
from sklearn.metrics.pairwise import pairwise_distances

# 计算用户相似度
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
print ("user_similarity: " + str(user_similarity))
# 计算物品相似度
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
print ("item_similarity: " + str(item_similarity))


# 预测
def predict(ratings, similarity, type='user'):
    # 基于用户相似度矩阵的
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    # 基于物品相似度矩阵
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# 预测结果
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')
print("item_prediction: " + str(item_prediction))
print("user_prediction: " + str(user_prediction))

# 评估指标，均方根误差
# 使用sklearn的mean_square_error (MSE)函数，其中，RMSE仅仅是MSE的平方根
# 只是想要考虑测试数据集中的预测评分，因此，使用pr
# ediction[ground_truth.nonzero()]筛选出预测矩阵中的所有其他元素
from sklearn.metrics import mean_squared_error, roc_auc_score
from math import sqrt


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# print(user_)
assert(user_prediction.shape == train_data_matrix.shape)
print ("user_prediction.shape")
print(user_prediction.shape)
print ("train_data_matrix.shape")
print(train_data_matrix.shape)
print('train-User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))


print('Test-User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))

print('``````````````````````````````````````````````````````````````````````')

print('train-Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))

print('Test-Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# 基于内存的CF的缺点是，它不能扩展到真实世界的场景，并且没有解决众所周知的冷启动问题，也就是当新用户或新产品进入
# 系统时。基于模型的CF方法是可扩展的，并且可以比基于内存的模型处理更高的稀疏度，但当没有任何评分的用户或产品进入
# 系统时，也是苦不堪言的
