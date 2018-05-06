# coding=utf-8
from scipy.sparse import csc_matrix
from sklearn.metrics import pairwise_distances
import numpy as np


from autoencoder.DenoisingAutoencoderModel import denoising_autoencoder_model
from base_cf.BaseCFModel import predict, base_cf
from evaluation.EvaluationModel import rmse, calculate_precision_recall, plot_precision_recall
from jaccard_cf.JaccardModel import jaccard_model
from python_redis.save_in_redis import MyRedis
from recommender.RecommenderModel import generate_top_recommendations, get_test_sample_recommendations
from song_model.DataModel import data_model
from svd_cf.SvdEvaluation import computeSVD, computeEstimatedRatings


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


def test_autoencoder_cf(data_model):
    auto = denoising_autoencoder_model(data_model.data_matrix, training_epochs = 50, model_path='./checkpoint_dir_song')
    auto.DAE_model()
    prediction = auto.evaluate_model()
    print("prediction")
    print(prediction)
    print("prediction.shape")
    print(prediction.shape)
    training_dict = generate_top_recommendations(data_model, prediction, 50)
    test_dict = get_test_sample_recommendations(data_model)
    item_ism_avg_recall_list, item_ism_avg_precision_list = calculate_precision_recall(data_model, training_dict,
                                                                                       test_dict,
                                                                                       top=50)
    redis = MyRedis()
    redis.setList("autoencoder_cf_model", item_ism_avg_recall_list, item_ism_avg_precision_list)
    plot_precision_recall(item_ism_avg_recall_list, item_ism_avg_precision_list, "autoencoder_cf_model")

def test_jaccard_cf(data_model):
    # m = data_model(sample_number=4000, test_size=0.4)
    jaccard = jaccard_model(data_model)
    cooccurence_matrix = jaccard.construct_cooccurence_matrix()

    # 推荐
    training_dict = generate_top_recommendations(data_model, cooccurence_matrix, top=50)
    test_dict = get_test_sample_recommendations(data_model)

    # 评估
    ism_avg_recall_list, ism_avg_precision_list = calculate_precision_recall(data_model, training_dict, test_dict, top=50)

    #保存到redis
    redis = MyRedis()
    redis.setList("jaccard_cf_model", ism_avg_recall_list, ism_avg_precision_list)
    plot_precision_recall(ism_avg_recall_list, ism_avg_precision_list, "jaccard_cf_model")

def test_svd_cf(data_model):
    K = 30
    train_matrix = data_model.data_matrix

    urm = csc_matrix(train_matrix, dtype=np.float)
    U, S, Vt = computeSVD(urm, K)
    prediction_matrix = computeEstimatedRatings(data_model, U, S, Vt)

    # 推荐
    training_dict = generate_top_recommendations(data_model, prediction_matrix, top=50)
    test_dict = get_test_sample_recommendations(data_model)

    # 评估
    ism_avg_recall_list, ism_avg_precision_list = calculate_precision_recall(data_model, training_dict, test_dict,
                                                                             top=50)

    redis = MyRedis()
    redis.setList("svd_cf_model", ism_avg_recall_list, ism_avg_precision_list)
    plot_precision_recall(ism_avg_recall_list, ism_avg_precision_list, "svd_cf_model")


data_model = data_model(sample_number=4000, test_size=0.4)
data_model.generate_data_matrix()
test_base_cf(data_model)
# test_autoencoder_cf(data_model)
# test_jaccard_cf(data_model)
# test_svd_cf(data_model)

#######
# redis = MyRedis()
# ism_avg_recall_list, ism_avg_precision_list = redis.getList("svd_cf_model")
# plot_precision_recall(ism_avg_recall_list, ism_avg_precision_list, "svd_cf_model")