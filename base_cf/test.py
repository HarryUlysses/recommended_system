# coding=utf-8
from sklearn.metrics import pairwise_distances
import numpy as np

from base_cf.BaseCFModel import predict
from evaluation.EvaluationModel import plot_precision_recall, calculate_precision_recall, rmse
from recommender.RecommenderModel import generate_top_recommendations, get_test_sample_recommendations
from song_model.DataModel import data_model

data_model = data_model(sample_number = 4000,test_size = 0.4)
data_model.generate_data_matrix()

# auto = denoising_autoencoder_model(data_model.data_matrix, training_epochs = 50, model_path='./checkpoint_dir_song')
# auto.DAE_model()
# prediction = auto.evaluate_model()
# print("prediction")
# print(prediction)
# print("prediction.shape")
# print(prediction.shape)

prediction = data_model.data_matrix

# 计算用户相似度
user_similarity = pairwise_distances(prediction, metric='cosine')
print ("user_similarity: " + str(user_similarity))
print ("user_similarity.shape" + str(user_similarity.shape))
print("````````````````````````````````````````````````")
print("")

# 计算物品相似度
item_similarity = pairwise_distances(prediction.T, metric='cosine')
print ("item_similarity: " + str(item_similarity))
print ("item_similarity: " + str(item_similarity.shape))
print("````````````````````````````````````````````````")
print("")

# 预测结果
item_prediction = predict(prediction, item_similarity, type='item')
user_prediction = predict(prediction, user_similarity, type='user')

print("item_prediction: " + str(item_prediction))
print("item_prediction.shape " + str(item_prediction.shape))
print("item_prediction: RMSE " + str(rmse(prediction, item_prediction)))
print("````````````````````````````````````````````````")
print("")

print("user_prediction: " + str(user_prediction))
print("user_prediction.shape: " + str(user_prediction.shape))
print("item_prediction: RMSE " + str(rmse(prediction, user_prediction)))
print("````````````````````````````````````````````````")
print("")

# 推荐
training_dict = generate_top_recommendations(data_model, item_prediction, 50)
test_dict = get_test_sample_recommendations(data_model)

# 评估
ism_avg_precision_list, ism_avg_recall_list = calculate_precision_recall(data_model, training_dict, test_dict,top=50)
print("max_precision:" + str(ism_avg_recall_list[np.array(ism_avg_recall_list).argsort()[-1]]))

plot_precision_recall(ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")

