import os
import numpy as np
import pandas as pd

import seaborn as sns




from sklearn.cluster import  KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import  euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

#read data

def get_data():
    data = pd.read_csv("../data/data.csv")
    return data

def get_genre_data():
    genre_data = pd.read_csv("../data/data_by_genres.csv")
    return genre_data

def get_year_data():
    year_data = pd.read_csv("../data/data_by_year.csv")
    return year_data


# print("-------INFO DATA-------")
# print(get_data().info())
# print("-------GENRE INFO DATA-------")
# print(get_genre_data().info())
# print("-------YEAR INFO DATA-------")
# print(get_year_data().info)

def get_decade(year):
    period_start = int(year/10) * 10
    decade = '{}s'.format(period_start)
    return decade

if __name__ == "__main__":
    get_data()
    get_genre_data()
    get_year_data()