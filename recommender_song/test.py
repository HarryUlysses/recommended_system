from DataModel import data_model
from JaccardModel import jaccard_model, plot_precision_recall

m = data_model(sample_number=1000)
j = jaccard_model(m)
j.construct_cooccurence_matrix()
j.generate_top_recommendations(top=10)
j.get_test_sample_recommendations()
ism_avg_precision_list, ism_avg_recall_list = j.calculate_precision_recall(top=10)
plot_precision_recall(ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
