import random
import pickle
import pandas as pd


# load data as table
def load_data(data_link, sep=',', header='infer', names=None):
    return pd.read_csv(open(data_link, 'r'), sep=sep, header=header, names=names)


# store object
def store_obj(obj, data_link):
    pickle.dump(obj, open(data_link, 'wb'), protocol=4)


# load object
def load_obj(data_link):
    return pickle.load(open(data_link, 'rb'))


# random choose a different
def random_choose_one(all_set, diff_set):
    choice = random.choice(all_set)
    while choice in diff_set:
        choice = random.choice(all_set)
    return choice
