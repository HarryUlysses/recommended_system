# coding=utf-8
import pandas
from sklearn import model_selection
import numpy as np

from autoencoder.DenoisingAutoencoderModel import denoising_autoencoder_model


class data_model:
    triplets_file = None
    songs_metadata_file = None

    song_df = None
    # data_matrix = None
    df_users = None
    df_songs = None

    train_data = None
    test_data = None

    train_users = None
    train_songs = None

    test_users = None
    test_songs = None
    ############################3
    data_matrix = None

    def __init__(self, triplets_file='../DataSets/song_recommender/10000.txt',
                 songs_metadata_file='../DataSets/song_recommender/song_data.csv',
                 sample_number=10000, test_size=0.30):
        self.triplets_file = triplets_file
        self.songs_metadata_file = songs_metadata_file
        # Read song  data
        song_df_1 = pandas.read_table(self.triplets_file, header=None)
        song_df_1.columns = ['user_id', 'song_id', 'listen_count']

        # Read song  metadata
        song_df_2 = pandas.read_csv(self.songs_metadata_file)

        # Merge the two dataframes above to create input dataframe for recommender systems
        song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

        # Create a subset of the dataset and initial variable
        self.song_df = song_df.head(sample_number)

        self.song_df['song'] = self.song_df['title'].map(str) + "-" + self.song_df['artist_name']

        # initial users & songs
        self.df_users = self.song_df['user_id'].unique()
        self.df_songs = self.song_df['song_id'].unique()

        self.train_data, self.test_data = model_selection.train_test_split(self.song_df, test_size=test_size,
                                                                           random_state=0)

        self.train_users = self.train_data['user_id'].unique()
        self.train_songs = self.train_data['song_id'].unique()

        self.test_users = self.test_data['user_id'].unique()
        self.test_songs = self.test_data['song_id'].unique()

        print ("song_df len: " + str(len(self.song_df)))
        print ("song_df_users : song_df_songs: " + str(len(self.df_users)) + " " + str(len(self.df_songs)))

        print ("train_data len: " + str(len(self.train_data)))
        print ("train_users : train_songs: " + str(len(self.train_users)) + " " + str(len(self.train_songs)))
        #
        print ("test_data len: " + str(len(self.test_data)))
        print ("test_users : test_songs: " + str(len(self.test_users)) + " " + str(len(self.test_songs)))
        print ("-----------------------------------------------------------------")

    def get_train_user_items(self, user):
        user_data = self.train_data[self.train_data['user_id'] == user]
        user_items = list(user_data['song_id'].unique())
        return user_items

        # Get unique users for a given item (song)

    def get_train_item_users(self, item):
        item_data = self.train_data[self.train_data['song_id'] == item]
        item_users = set(item_data['user_id'].unique())
        return item_users

    def get_train_user_item_listen_number(self, user_id, item_id):
        items_data = self.train_data[self.train_data['user_id'] == user_id]
        item_data = items_data[items_data['song_id'] == item_id]
        assert isinstance(item_data, object)
        return item_data

    def generate_data_matrix(self):
        self.data_matrix = np.zeros(shape=(len(self.train_users), len(self.train_songs)), dtype=float)
        # self.data_matrix = np.matrix(np.zeros(shape=(len(self.users), len(self.songs)), float))
        # print ("matrix.shape " + str(self.data_matrix.shape))
        # userIndex = list(self.users)
        # songIndex = list(self.songs)
        # for index in range(0,len(userIndex)):
        #     print("key,value"+ str(index) +" "+ str(userIndex[index]))

        # print(len(userIndex))
        for i in range(0, len(self.train_users)):
            # songs_number_sum = self.get_user_song_number_sum(userIndex[i])
            user_items = self.get_train_user_items(self.train_users[i])
            for j in range(0, len(user_items)):
                song_select = self.get_train_user_item_listen_number(self.train_users[i], user_items[j])

                self.data_matrix[i][j] = song_select['listen_count'].sum()
        self.data_matrix = 2. / (
                    1 + np.exp(-1. * (self.data_matrix))) - 1
        print ("orign_train_data.shape" + str(self.data_matrix.shape))
        # print ("orign_train_data.type" + str(type(self.data_matrix)))
        # train_data, test_data = model_selection.train_test_split(self.data_matrix, test_size=0.20, random_state=0)
        # print ("train_data.shape" + str(train_data.shape))
        # print ("test_data.shape" + str(test_data.shape))
        # print (self.data_matrix)
        # print (self.data_matrix.sum(axis=1))
        # print (len(self.data_matrix.sum(axis=1)))
        print ("-----------------------------------------------------------------")
