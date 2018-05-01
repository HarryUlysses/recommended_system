# coding=utf-8
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

from DenoisingAutoencoderModel import denoising_autoencoder_model

cols = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
        '13', '14', '15', '16', '17', '18', '19', '20', '21', '22',
        '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']

df = pd.read_csv("./groceries.csv", sep=",",
                 names=cols, engine="python")
data = np.array(df)

def get_unique_items(data):
    ncol = data.shape[1]
    items = set()
    for c in range(ncol):
        items = items.union(data[:, c])
    items = np.array(list(items))
    items = items[items != np.array(None)]
    return np.unique(items)


def get_onehot_items(data, unique_items):
    onehot_items = np.zeros((len(data), len(unique_items)), dtype=np.int)
    for i, r in enumerate(data):
        for j, c in enumerate(unique_items):
            onehot_items[i, j] = int(c in r)

    return onehot_items


def get_items_from_ohe(ohe, unique_items):
    return unique_items[np.flatnonzero(ohe)]


unique_items = get_unique_items(data)
onehot_items = np.array(get_onehot_items(data, unique_items))


model = denoising_autoencoder_model(onehot_items)

model.evaluate_model()
# model.DAE_model()
