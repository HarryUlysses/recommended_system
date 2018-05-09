# coding=utf-8
import math as mt
from sparsesvd import sparsesvd  # used for matrix factorization
import numpy as np
from scipy.sparse import csc_matrix  # used for sparse matrix
from scipy.sparse.linalg import *  # used for matrix multiplication
from sklearn.metrics import pairwise_distances

from base_cf.BaseCFModel import predict
from evaluation.EvaluationModel import calculate_precision_recall, plot_precision_recall
from recommender.RecommenderModel import generate_top_recommendations, get_test_sample_recommendations
from song_model.DataModel import data_model



# Compute SVD of the user ratings matrix
def computeSVD(urm, K):
    U, s, Vt = sparsesvd(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = np.transpose(U)
    return U, S, Vt


# Compute estimated rating for the test user
def computeEstimatedRatings(data_model, U, S, Vt):
    rightTerm = U.dot(S.dot(Vt))
    return rightTerm


# data_model = data_model(sample_number=4000, test_size=0.4)
# data_model.generate_data_matrix()
# K = 30
# train_matrix = data_model.data_matrix
#
# urm = csc_matrix(train_matrix, dtype=np.float)
# U, S, Vt = computeSVD(urm, K)
# prediction_matrix = computeEstimatedRatings(data_model, U, S, Vt)
# print (prediction_matrix)
# print (prediction_matrix.shape)


# # 计算用户相似度
# user_similarity = pairwise_distances(prediction_matrix, metric='cosine')
# print ("user_similarity: " + str(user_similarity))
# print ("user_similarity.shape" + str(user_similarity.shape))
# print("````````````````````````````````````````````````")
# print("")
#
# # 计算物品相似度
# item_similarity = pairwise_distances(prediction_matrix.T, metric='cosine')
# print ("item_similarity: " + str(item_similarity))
# print ("item_similarity: " + str(item_similarity.shape))
# print("````````````````````````````````````````````````")
# print("")
#
# # 预测结果
# item_prediction = predict(prediction_matrix, item_similarity, type='item')
# user_prediction = predict(prediction_matrix, user_similarity, type='user')
#
# print("item_prediction: " + str(item_prediction))
# print("item_prediction.shape " + str(item_prediction.shape))
# print("````````````````````````````````````````````````")
# print("")
#
# print("user_prediction: " + str(user_prediction))
# print("user_prediction.shape: " + str(user_prediction.shape))
# print("````````````````````````````````````````````````")
# print("")





# 推荐
# training_dict = generate_top_recommendations(data_model, prediction_matrix, top = 50)
# test_dict = get_test_sample_recommendations(data_model)
#
# # 评估
# ism_avg_precision_list, ism_avg_recall_list = calculate_precision_recall(data_model, training_dict, test_dict,top=50)
# print("max_precision:" + str(ism_avg_recall_list[np.array(ism_avg_recall_list).argsort()[-1]]))
#
# plot_precision_recall(ism_avg_precision_list, ism_avg_recall_list, "svd_model")