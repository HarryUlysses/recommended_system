# # coding=utf-8
# import pandas
# from sklearn import model_selection
# import numpy as np
# import time
# from sklearn.externals import joblib
# import Recommenders as Recommenders
# import Evaluation as Evaluation
# import pylab as pl
# ##开始
# triplets_file = '../DataSets/song_recommender/10000.txt'
# songs_metadata_file = '../DataSets/song_recommender/song_data.csv'
#
# song_df_1 = pandas.read_table(triplets_file, header=None)
# song_df_1.columns = ['user_id', 'song_id', 'listen_count']
#
# #Read song  metadata
# song_df_2 =  pandas.read_csv(songs_metadata_file)
#
# #Merge the two dataframes above to create input dataframe for recommender systems
# song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
#
#
# #Create a subset of the dataset
# song_df = song_df.head(5000)
# #Merge song title and artist_name columns to make a merged column
# song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']
# song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
#
# song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
# grouped_sum = song_grouped['listen_count'].sum()
# song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
# song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
#
# users = song_df['user_id'].unique()
# print ("Userlen: "+str(len(users)))
# songs = song_df['song'].unique()
# print ("Userlen: "+str(len(songs)))
#
# train_data, test_data = model_selection.train_test_split(song_df, test_size = 0.20, random_state=0)
# print(train_data.head(5))
#
# pm = Recommenders.popularity_recommender_py()
# pm.create(train_data, 'user_id', 'song')
#
# user_id = users[5]
# pm.recommend(user_id)
#
# is_model = Recommenders.item_similarity_recommender_py()
# is_model.create(train_data, 'user_id', 'song')
#
# #Print the songs for the user in training data
# user_id = users[5]
# user_items = is_model.get_user_items(user_id)
# #
# print("------------------------------------------------------------------------------------")
# print("Training data songs for the user userid: %s:" % user_id)
# print("------------------------------------------------------------------------------------")
#
# for user_item in user_items:
#     print(user_item)
#
# print("----------------------------------------------------------------------")
# print("Recommendation process going on:")
# print("----------------------------------------------------------------------")
#
# #Recommend songs for the user using personalized model
# is_model.recommend(user_id)
#
# start = time.time()
#
# #Define what percentage of users to use for precision recall calculation
# user_sample = 0.05
#
# #Instantiate the precision_recall_calculator class
# pr = Evaluation.precision_recall_calculator(test_data, train_data, pm, is_model)
#
# #Call method to calculate precision and recall values
# (pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)
#
# end = time.time()
# print(end - start)
#
#
#
# #Method to generate precision and recall curve
# def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m2_recall_list, m2_label):
#     pl.clf()
#     pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
#     pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
#     pl.xlabel('Recall')
#     pl.ylabel('Precision')
#     pl.ylim([0.0, 0.20])
#     pl.xlim([0.0, 0.20])
#     pl.title('Precision-Recall curve')
#     #pl.legend(loc="upper right")
#     pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
#     pl.show()
#
# print("Plotting precision recall curves.")
#
# plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
#                       ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")