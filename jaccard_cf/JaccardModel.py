# coding= utf-8
import numpy as np


class jaccard_model:
    data_model = None
    concurrence_matrix = None

    def __init__(self, data_model):
        self.data_model = data_model

    def construct_cooccurence_matrix(self):

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
                        # print(user_cooccurence_matrix[j, i])
                    else:
                        user_cooccurence_matrix[j, i] = 0
            user_sim_scores = user_cooccurence_matrix.sum(axis=0) / float(user_cooccurence_matrix.shape[0])
            user_sim_scores = np.array(user_sim_scores).tolist()
            cooccurence.append(user_sim_scores)
        self.concurrence_matrix = np.array(cooccurence)
        print("concurrence_matrix: ")
        print(self.concurrence_matrix)
        print(self.concurrence_matrix.shape)
        return self.concurrence_matrix

