# coding=utf-8
from sklearn.metrics import pairwise_distances
import numpy as np

from python_redis.save_in_redis import MyRedis
from song_model.DataModel import data_model
from base_cf.BaseCFModel import predict, base_cf
from evaluation.EvaluationModel import plot_precision_recall, calculate_precision_recall, rmse
from recommender.RecommenderModel import generate_top_recommendations, get_test_sample_recommendations
from song_model.DataModel import data_model


def test_base_cf(data_model):
    item_prediction, user_prediction = base_cf(data_model.data_matrix)
    item_rmse = rmse(item_prediction, data_model.data_matrix)
    user_rmse = rmse(user_prediction, data_model.data_matrix)

    # item 推荐
    item_training_dict = generate_top_recommendations(data_model, item_prediction, 50)
    item_test_dict = get_test_sample_recommendations(data_model)

    # user 推荐
    user_training_dict = generate_top_recommendations(data_model, user_prediction, 50)
    user_test_dict = get_test_sample_recommendations(data_model)

    # 评估
    item_ism_avg_recall_list, item_ism_avg_precision_list = calculate_precision_recall(data_model, item_training_dict,
                                                                                       item_test_dict,
                                                                                       top=50)
    user_ism_avg_recall_list, user_ism_avg_precision_list = calculate_precision_recall(data_model, user_training_dict,
                                                                                       user_test_dict,
                                                                                       top=50)
    redis = MyRedis()
    redis.setList("base_cf_cos_item",item_ism_avg_recall_list,item_ism_avg_precision_list)
    redis.setList("base_cf_cos_user", user_ism_avg_recall_list, user_ism_avg_precision_list)
    plot_precision_recall(item_ism_avg_recall_list, item_ism_avg_precision_list, "base_cf_cos_item",user_ism_avg_recall_list,user_ism_avg_precision_list,"base_cf_cos_user")
    return item_rmse, user_rmse
