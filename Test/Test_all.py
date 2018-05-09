# coding=utf-8
from scipy.sparse import csc_matrix
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
    # user_rmse = rmse(user_prediction, data_model.data_matrix)

    # item 推荐
    item_training_dict = generate_top_recommendations(data_model, item_prediction, 50)
    item_test_dict = get_test_sample_recommendations(data_model)

    # # user 推荐
    # user_training_dict = generate_top_recommendations(data_model, user_prediction, 50)
    # user_test_dict = get_test_sample_recommendations(data_model)

    # 评估
    item_ism_avg_recall_list, item_ism_avg_precision_list = calculate_precision_recall(data_model, item_training_dict,
                                                                                       item_test_dict,
                                                                                       top=50)
    # user_ism_avg_recall_list, user_ism_avg_precision_list = calculate_precision_recall(data_model, user_training_dict,
    #                                                                                    user_test_dict,
    #                                                                                    top=50)
    redis = MyRedis()
    redis.setList("base_cf_cos_item",item_ism_avg_recall_list,item_ism_avg_precision_list)
    redis.setRmse("base_cf_cos_item_Rmse", item_rmse)
    # redis.setList("base_cf_cos_user", user_ism_avg_recall_list, user_ism_avg_precision_list)
    plot_precision_recall(item_ism_avg_recall_list, item_ism_avg_precision_list, "base_cf_cos_item")

def test_autoencoder_item_cf(data_model):
    item_prediction, user_prediction = base_cf(data_model.data_matrix)
    item_rmse = rmse(item_prediction, data_model.data_matrix)
    # user_rmse = rmse(user_prediction, data_model.data_matrix)

    auto = denoising_autoencoder_model(item_prediction, training_epochs=1000, model_path='./checkpoint_dir_song')
    auto.DAE_model()
    prediction, autoencoder_item_RMSE = auto.evaluate_model()

    # item 推荐
    item_training_dict = generate_top_recommendations(data_model, prediction, 50)
    item_test_dict = get_test_sample_recommendations(data_model)


    # 评估
    item_ism_avg_recall_list, item_ism_avg_precision_list = calculate_precision_recall(data_model, item_training_dict,
                                                                                       item_test_dict,
                                                                                       top=50)
    redis = MyRedis()
    redis.setList("base_autoencoder__item_cf", item_ism_avg_recall_list, item_ism_avg_precision_list)
    redis.setRmse("base_autoencoder__item_cf_Rmse", autoencoder_item_RMSE)
    plot_precision_recall(item_ism_avg_recall_list, item_ism_avg_precision_list, "base_autoencoder__item_cf")



def test_autoencoder_cf(data_model):
    auto = denoising_autoencoder_model(data_model.data_matrix, training_epochs = 1000, model_path='./checkpoint_dir_song')
    auto.DAE_model()
    prediction,autoencoder_cf_Rmse = auto.evaluate_model()
    #推荐
    training_dict = generate_top_recommendations(data_model, prediction, 50)
    test_dict = get_test_sample_recommendations(data_model)
    item_ism_avg_recall_list, item_ism_avg_precision_list = calculate_precision_recall(data_model, training_dict,
                                                                                       test_dict,
                                                                                       top=50)
    redis = MyRedis()
    redis.setList("autoencoder_cf_model", item_ism_avg_recall_list, item_ism_avg_precision_list)
    redis.setRmse("autoencoder_cf_model_Rmse", autoencoder_cf_Rmse)
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
    K = 100
    train_matrix = data_model.data_matrix

    urm = csc_matrix(train_matrix, dtype=np.float)
    U, S, Vt = computeSVD(urm, K)
    prediction_matrix = computeEstimatedRatings(data_model, U, S, Vt)
    svd_Rmse = rmse(train_matrix,prediction_matrix)
    # 推荐
    training_dict = generate_top_recommendations(data_model, prediction_matrix, top=50)
    test_dict = get_test_sample_recommendations(data_model)

    # 评估
    ism_avg_recall_list, ism_avg_precision_list = calculate_precision_recall(data_model, training_dict, test_dict,
                                                                             top=50)

    redis = MyRedis()
    redis.setList("svd_cf_model", ism_avg_recall_list, ism_avg_precision_list)
    redis.setRmse("svd_cf_model_Rmse", svd_Rmse)
    plot_precision_recall(ism_avg_recall_list, ism_avg_precision_list, "svd_cf_model")



redis = MyRedis()
base_cf_cos_item_Rmse = redis.getRmse("base_cf_cos_item_Rmse")
base_autoencoder__item_cf_Rmse = redis.getRmse("base_autoencoder__item_cf_Rmse")
autoencoder_cf_model_Rmse = redis.getRmse("autoencoder_cf_model_Rmse")
svd_cf_model_Rmse = redis.getRmse("svd_cf_model_Rmse")
print("base_cf_cos_item_Rmse:"+str(base_cf_cos_item_Rmse))
print("base_autoencoder__item_cf_Rmse:"+str(base_autoencoder__item_cf_Rmse))
print("autoencoder_cf_model_Rmse:"+str(autoencoder_cf_model_Rmse))
print("svd_cf_model_Rmse:"+str(svd_cf_model_Rmse))


