import pandas as pd
import os
from os import path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def plot_len_dist(texts):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.hist([len(l) for l in texts], bins=100)
    plt.savefig(CONFIG['path']['plot']['dataset_length_dist'])

def get_90percentile_length(texts):
    return np.percentile([len(l) for l in texts], 90)


def merge_data():
    df_list = []
    files = os.listdir(CONFIG['path']['data_files'])
    for f in files:
        dff = pd.read_csv(path.join(CONFIG['path']['data_files'], f), index_col=False)
        df_list.append(dff)
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['text']).iloc[:, [0] + [i for i in range(9, len(df.columns))]]
    df.fillna(0, inplace=True)
    df = df.loc[df.iloc[:, 1:].sum(axis=1) != 0]
    df = df.replace({'’': "'"}, regex=True)
    return df

def load_data():
    df = merge_data()
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1:]
    train_texts, test_texts, train_labels, test_labels =\
         train_test_split(texts, labels, random_state=42, test_size=0.3)
    test_texts, val_texts, test_labels, val_labels =\
        train_test_split(test_texts, test_labels, random_state=42, test_size=0.5)

    return (
        (train_texts.values, train_labels.values),
        (val_texts.values, val_labels.values),
        (test_texts.values, test_labels.values)
    )

def export_dataset():
    df = merge_data()
    df.to_csv(CONFIG['path']['dataset'], index=False)