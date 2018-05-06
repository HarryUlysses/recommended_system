import pandas
import numpy as np

def generate_top_recommendations(data_model, concurrence_matrix, top=10):
    # Sort the indices of user_sim_scores based upon their value
    # Also maintain the corresponding score
    training_dict = dict()
    for user_index in range(0, len(concurrence_matrix)):
        #
        user_id = data_model.train_users[user_index]
        user_sim_scores = concurrence_matrix[user_index]

        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)
        # print("sort_index:" + str(sort_index))

        # Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)

        user_songs = data_model.get_train_user_items(user_id)

        # Fill the dataframe with top 10 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and data_model.train_songs[sort_index[i][1]] not in user_songs \
                    and rank <= top:
                df.loc[len(df)] = [user_id, data_model.train_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")

        training_dict[user_id] = list(df["song"])
    # print("train_dict")
    # print(training_dict)
    return training_dict



def get_test_sample_recommendations(data_model):
    # For these test_sample users, get top 10 recommendations from training set
    # self.ism_training_dict = {}
    # self.pm_training_dict = {}

    test_dict = {}
    # users_test_and_training = list(
    #     set(self.test_data['user_id'].unique()).intersection(set(self.train_data['user_id'].unique())))
    for user_id in data_model.test_users:
        # Get items for user_id from test_data
        test_data_user = data_model.test_data[data_model.test_data['user_id'] == user_id]
        test_dict[user_id] = set(test_data_user['song_id'].unique())
    # print("test_dict")
    # print(test_dict)
    return test_dict

