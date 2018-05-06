# coding=utf-8

from DataModel import data_model
from autoencoder.DenoisingAutoencoderModel import denoising_autoencoder_model
from evaluation.EvaluationModel import calculate_precision_recall, plot_precision_recall
from jaccard_cf.JaccardModel import jaccard_model
from recommender.RecommenderModel import generate_top_recommendations, get_test_sample_recommendations

m = data_model(sample_number=4000,test_size = 0.4)
j = jaccard_model(m)
cooccurence_matrix = j.construct_cooccurence_matrix()

# 推荐
training_dict = generate_top_recommendations(m, cooccurence_matrix, top=50)
test_dict = get_test_sample_recommendations(m)

# 评估
ism_avg_precision_list, ism_avg_recall_list = calculate_precision_recall(m, training_dict, test_dict, top=50)
plot_precision_recall(ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")

# a = data_model(sample_number = 10000)
# a.generate_data_matrix()
# auto = denoising_autoencoder_model(a.data_matrix, training_epochs = 70, model_path='./checkpoint_dir_song')
# auto.DAE_model()
# prediction = auto.evaluate_model()
# print("prediction.shape")
# print(prediction.shape)

## 推荐
# training_dict = generate_top_recommendations(a, prediction,50)
# test_dict = get_test_sample_recommendations(a)

## 评估
# ism_avg_precision_list, ism_avg_recall_list = calculate_precision_recall(a, training_dict, test_dict,top=50)
# plot_precision_recall(ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
