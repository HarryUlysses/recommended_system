# coding= utf-8
import numpy as np
import pandas
import pylab as pl


def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label):
    # , m2_precision_list, m2_recall_list, m2_label
    pl.clf()
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    # pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
    pl.show()


class jaccard_model:
    data_model = None
    concurrence_matrix = None
    training_dict = dict()
    test_dict = {}

    def __init__(self, data_model):
        self.data_model = data_model

    def construct_cooccurence_matrix(self):

        ####################################
        # Get users for all songs in user_songs.
        ####################################
        users_songs_users = []
        for user in self.data_model.train_users:
            user_songs_users = []
            user_songs = self.data_model.get_train_user_items(user)
            for song in user_songs:
                user_songs_users.append(self.data_model.get_train_item_users(song))
            users_songs_users.append(user_songs_users)

        ###############################################
        # Initialize the item cooccurence matrix of size
        # len(user_songs) X len(songs)
        ###############################################
        cooccurence = []
        for user_index in range(0, len(self.data_model.train_users)):
            user_songs = self.data_model.get_train_user_items(self.data_model.train_users[user_index])

            user_cooccurence_matrix = np.zeros(shape=(len(user_songs), len(self.data_model.train_songs)), dtype=float)

            for i in range(0, len(self.data_model.train_songs)):
                # Calculate unique listeners (users) of song (item) i
                users_i = self.data_model.get_train_item_users(self.data_model.train_songs[i])

                for j in range(0, len(user_songs)):

                    # Get unique listeners (users) of song (item) j
                    # 听过这首歌的所有人数
                    users_j = users_songs_users[user_index][j]

                    # Calculate intersection of listeners of songs i and j
                    users_intersection = users_i.intersection(users_j)

                    # Calculate cooccurence_matrix[i,j] as Jaccard Index
                    if len(users_intersection) != 0:
                        # Calculate union of listeners of songs i and j
                        users_union = users_i.union(users_j)

                        user_cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                    else:
                        user_cooccurence_matrix[j, i] = 0
            user_sim_scores = user_cooccurence_matrix.sum(axis=0) / float(user_cooccurence_matrix.shape[0])
            user_sim_scores = np.array(user_sim_scores).tolist()
            cooccurence.append(user_sim_scores)
        self.concurrence_matrix = np.array(cooccurence)
        # print("concurrence_matrix: ")
        # print(self.concurrence_matrix)
        # print(self.concurrence_matrix.shape)

    def generate_top_recommendations(self, top=10):
        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        for user_index in range(0, len(self.concurrence_matrix)):
            #

            user_id = self.data_model.train_users[user_index]
            user_sim_scores = self.concurrence_matrix[user_index]

            sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)
            # print("sort_index:" + str(sort_index))

            # Create a dataframe from the following
            columns = ['user_id', 'song', 'score', 'rank']
            # index = np.arange(1) # array of numbers for the number of samples
            df = pandas.DataFrame(columns=columns)

            user_songs = self.data_model.get_train_user_items(user_id)

            # Fill the dataframe with top 10 item based recommendations
            rank = 1
            for i in range(0, len(sort_index)):
                if ~np.isnan(sort_index[i][0]) and self.data_model.train_songs[sort_index[i][1]] not in user_songs \
                        and rank <= top:
                    df.loc[len(df)] = [user_id, self.data_model.train_songs[sort_index[i][1]], sort_index[i][0], rank]
                    rank = rank + 1

            # Handle the case where there are no recommendations
            if df.shape[0] == 0:
                print("The current user has no songs for training the item similarity based recommendation model.")

            self.training_dict[user_id] = list(df["song"])
        # print("train_dict")
        # print(self.training_dict)

    def get_test_sample_recommendations(self):
        # For these test_sample users, get top 10 recommendations from training set
        # self.ism_training_dict = {}
        # self.pm_training_dict = {}

        # self.test_dict = {}
        # users_test_and_training = list(
        #     set(self.test_data['user_id'].unique()).intersection(set(self.train_data['user_id'].unique())))
        for user_id in self.data_model.test_users:
            # Get items for user_id from test_data
            test_data_user = self.data_model.test_data[self.data_model.test_data['user_id'] == user_id]
            self.test_dict[user_id] = set(test_data_user['song_id'].unique())
        # print("test_dict")
        # print(self.test_dict)

    # Method to calculate the precision and recall measures
    def calculate_precision_recall(self, top=10):
        # Create cutoff list for precision and recall calculation
        cutoff_list = list(range(1, top + 1))

        # For each distinct cutoff:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.

        ism_avg_precision_list = []
        ism_avg_recall_list = []

        num_test_users_sample = len(self.data_model.test_users)

        for N in cutoff_list:
            ism_sum_precision = 0
            ism_sum_recall = 0

            for user_id in self.data_model.test_users:
                if user_id in self.training_dict.keys():
                    ism_hitset = self.test_dict[user_id].intersection(set(self.training_dict[user_id][0:N]))
                    testset = self.test_dict[user_id]

                    ism_sum_precision += float(len(ism_hitset)) / float(len(testset))
                    ism_sum_recall += float(len(ism_hitset)) / float(N)
            ism_avg_precision = ism_sum_precision / float(num_test_users_sample)
            ism_avg_recall = ism_sum_recall / float(num_test_users_sample)

            ism_avg_precision_list.append(ism_avg_precision)
            ism_avg_recall_list.append(ism_avg_recall)

        return ism_avg_precision_list, ism_avg_recall_list
