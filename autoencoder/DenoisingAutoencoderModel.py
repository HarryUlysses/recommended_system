# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from math import sqrt
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns


# sns.set_style("whitegrid")

class denoising_autoencoder_model:
    train_data = None
    valid_data = None
    test_data = None

    input_dim = None

    n_hidden_1 = 128
    n_hidden_2 = 64
    n_hidden_3 = 32
    n_hidden_4 = 16

    training_epochs = 30
    batch_size = 10
    total_batches = None

    learning_rate = 0.00002
    keep_prob = 0.6
    l2_reg_rate = 0.00001

    ############################
    e_weights_h1 = None
    e_biases_h1 = None

    e_weights_h2 = None
    e_biases_h2 = None

    e_weights_h3 = None
    e_biases_h3 = None

    e_weights_h4 = None
    e_biases_h4 = None

    d_weights_h1 = None
    d_biases_h1 = None

    d_weights_h2 = None
    d_biases_h2 = None

    d_weights_h3 = None
    d_biases_h3 = None

    d_weights_h4 = None
    d_biases_h4 = None

    model_path = None
    model_name = None

    def __init__(self, train_data, thresholds=0.8, training_epochs=30, batch_size=10, learning_rate=0.00002,
                 keep_prob=0.6, l2_reg_rate=0.00001, model_path='./checkpoint_dir', model_name = 'MyModel'):
        self.model_path = model_path
        self.model_name = model_name


        train_test_split = np.random.rand(len(train_data)) < thresholds
        self.test_data = train_data[~train_test_split]
        self.train_data = train_data[train_test_split]

        train_validation_split = np.random.rand(len(self.train_data)) < thresholds
        self.valid_data = self.train_data[~train_validation_split]
        self.train_data = self.train_data[train_validation_split]

        self.input_dim = train_data.shape[1]
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.total_batches = (self.train_data.shape[0] // self.batch_size)
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.l2_reg_rate = l2_reg_rate

    def weight_variable(self, weight_name, weight_shape):
        return tf.get_variable(name="weight_" + weight_name, shape=weight_shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, bias_shape):
        initial = tf.constant(0.1, shape=bias_shape)
        return tf.Variable(initial)

    def encoder_variables(self):
        # --------------------- Encoder Variables --------------- #
        self.e_weights_h1 = self.weight_variable("el1", [self.input_dim, self.n_hidden_1])
        self.e_biases_h1 = self.bias_variable([self.n_hidden_1])

        self.e_weights_h2 = self.weight_variable("el2", [self.n_hidden_1, self.n_hidden_2])
        self.e_biases_h2 = self.bias_variable([self.n_hidden_2])

        self.e_weights_h3 = self.weight_variable("el3", [self.n_hidden_2, self.n_hidden_3])
        self.e_biases_h3 = self.bias_variable([self.n_hidden_3])

        self.e_weights_h4 = self.weight_variable("el4", [self.n_hidden_3, self.n_hidden_4])
        self.e_biases_h4 = self.bias_variable([self.n_hidden_4])

    def decoder_variables(self):
        # --------------------- Decoder Variables --------------- #
        self.d_weights_h1 = self.weight_variable("dl1", [self.n_hidden_4, self.n_hidden_3])
        self.d_biases_h1 = self.bias_variable([self.n_hidden_3])

        self.d_weights_h2 = self.weight_variable("dl2", [self.n_hidden_3, self.n_hidden_2])
        self.d_biases_h2 = self.bias_variable([self.n_hidden_2])

        self.d_weights_h3 = self.weight_variable("dl3", [self.n_hidden_2, self.n_hidden_1])
        self.d_biases_h3 = self.bias_variable([self.n_hidden_1])

        self.d_weights_h4 = self.weight_variable("dl4", [self.n_hidden_1, self.input_dim])
        self.d_biases_h4 = self.bias_variable([self.input_dim])

    def encoder(self, x):
        l1 = tf.nn.softsign(tf.add(tf.matmul(x, self.e_weights_h1), self.e_biases_h1))
        l2 = tf.nn.softsign(tf.add(tf.matmul(l1, self.e_weights_h2), self.e_biases_h2))
        l3 = tf.nn.softsign(tf.add(tf.matmul(l2, self.e_weights_h3), self.e_biases_h3))
        l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, self.e_weights_h4), self.e_biases_h4))
        return l4

    def decoder(self, x):
        l1 = tf.nn.softsign(tf.add(tf.matmul(x, self.d_weights_h1), self.d_biases_h1))
        l2 = tf.nn.softsign(tf.add(tf.matmul(l1, self.d_weights_h2), self.d_biases_h2))
        l3 = tf.nn.softsign(tf.add(tf.matmul(l2, self.d_weights_h3), self.d_biases_h3))
        l4 = tf.nn.sigmoid(tf.add(tf.matmul(l3, self.d_weights_h4), self.d_biases_h4))
        return l4

    def training_DAE_model(self, X, is_training, optimizer, cost_function, decoded):
        with tf.Session() as session:

            tf.global_variables_initializer().run()
            print("Epoch", "  ", "Tr. Loss", " ", "Val. Loss")

            for epoch in range(self.training_epochs):
                for b in range(self.total_batches):
                    offset = (b * self.batch_size) % (self.train_data.shape[0] - self.batch_size)
                    baseset = ((b * self.batch_size) // (self.train_data.shape[0] - self.batch_size)) * (
                            self.train_data.shape[0] - self.batch_size)
                    batch_x = self.train_data[baseset + offset:(baseset + offset + self.batch_size), :]
                    _, c = session.run([optimizer, cost_function], feed_dict={X: batch_x, is_training: True})

                tr_c = session.run(cost_function, feed_dict={X: self.train_data, is_training: False})
                val_c = session.run(cost_function, feed_dict={X: self.valid_data, is_training: False})
                print(epoch, " ", tr_c, " ", val_c)
            tf.add_to_collection('decoded', decoded)
            saver = tf.train.Saver()
            saver.save(session, self.model_path)

    def rmse(self, prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))

    def evaluate_model(self):
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(self.model_path + '/' + self.model_name + '.meta')
            saver.restore(session, tf.train.latest_checkpoint(self.model_path))

            decoded = tf.get_collection('decoded')[0]
            # get variable from tensor
            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            is_training = graph.get_tensor_by_name("is_training:0")

            tr_p = session.run(decoded, feed_dict={X: self.train_data, is_training: False})
            roc_auc = roc_auc_score(self.train_data, tr_p, average="samples")
            print("traingData.shape")
            print(self.train_data.shape)
            print("tr_pData.shape")
            print(tr_p.shape)
            print("Training ROC AUC: ", round(roc_auc, 4))
            print("Training RMSE: ", self.rmse(tr_p, self.train_data))

            print("train_data: ")
            print(self.train_data)

            print("tr_pData: ")
            print(tr_p)

            # val_p = session.run(decoded, feed_dict={X: validation_x, is_training: False})
            # roc_auc = roc_auc_score(validation_x, val_p, average="samples")
            # print("Validation ROC AUC: ", round(roc_auc, 4))
            # print("Validation RMSE: ", rmse(val_p, validation_x))
            #
            # ts_p = session.run(decoded, feed_dict={X: test_x, is_training: False})
            # roc_auc = roc_auc_score(test_x, ts_p, average="samples")
            # print("Test ROC AUC: ", round(roc_auc, 4), "\n")
            # print("Test RMSE: ", rmse(ts_p, test_x))
            #
            # # ----------------------------------------------------------------
            # item_preds = session.run(decoded, feed_dict={X: test_x.reshape(-1, 169), is_training: False})
            # item_preds[item_preds >= 0.1] = 1
            # item_preds[item_preds < 0.1] = 0
            # -------------------------------------------------------------------------------- #

    def DAE_model(self):
        tf.reset_default_graph()
        # 初始化参数
        is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
        X = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='X')
        X_drop = tf.contrib.layers.dropout(X, self.keep_prob, is_training=is_training)
        #
        self.encoder_variables()
        self.decoder_variables()
        encoded = self.encoder(X_drop)
        decoded = self.decoder(encoded)

        regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg_rate)
        reg_loss = regularizer(self.e_weights_h1) + regularizer(self.e_weights_h2) + regularizer(
            self.e_weights_h3) + regularizer(
            self.e_weights_h4)
        cost_function = -tf.reduce_mean(((X * tf.log(decoded)) + ((1 - X) * tf.log(1 - decoded)))) + reg_loss
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost_function)
        self.training_DAE_model(X, is_training, optimizer, cost_function, decoded)
        # self.evaluate_model()

# a = np.array([[1,2],[3,4]])
# b = denoising_autoencoder_model(a,learning_rate=1)
# print (b.learning_rate)
