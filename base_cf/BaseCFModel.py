# coding=utf-8
# 计算相似度
# 使用sklearn的pairwise_distances函数来计算余弦相似性
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error, roc_auc_score
from math import sqrt
import numpy as np

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
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=0)])
    return pred

def base_cf(data_matrix):
    # 计算用户相似度
    user_similarity = pairwise_distances(data_matrix, metric='cosine')
    # 计算物品相似度
    item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

    # 预测结果
    item_prediction = predict(data_matrix, item_similarity, type='item')
    user_prediction = predict(data_matrix, user_similarity, type='user')

    return item_prediction, user_prediction


