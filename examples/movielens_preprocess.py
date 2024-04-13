# -*- coding: utf-8 -*-
"""
==============================
prerpocess MovieLens dataset
==============================

In this script, we pre-process the MovieLens 10M Dataset to get the right format of contextual bandit
algorithms. This data set is released by GroupLens at 1/2009. Please fist download the dataset from
http://grouplens.org/datasets/movielens/, then unzipped the file 'ml-1m.zip' to the examples folder.
"""

import pandas as pd
import numpy as np
import itertools

from os.path import join

DATASET_NAME = 'ml-1m'
DATA_DIR = join('data', DATASET_NAME)
RAW_DATA_DIR = join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = join(DATA_DIR, 'processed')

def movie_preprocessing(movie):
    movie_col = list(movie.columns)
    movie_tag = [doc.split('|') for doc in movie['tag']]
    tag_table = {token: idx for idx, token in enumerate(set(itertools.chain.from_iterable(movie_tag)))}
    movie_tag = pd.DataFrame(movie_tag)
    tag_table = pd.DataFrame(tag_table.items())
    tag_table.columns = ['Tag', 'Index']

    # use one-hot encoding for movie genres (here called tag)
    tag_dummy = np.zeros([len(movie), len(tag_table)])

    for i in range(len(movie)):
        for j in range(len(tag_table)):
            if tag_table['Tag'][j] in list(movie_tag.iloc[i, :]):
                tag_dummy[i, j] = 1

    # combine the tag_dummy one-hot encoding table to original movie files
    movie = pd.concat([movie, pd.DataFrame(tag_dummy)], axis=1)
    movie_col.extend(['Tag:' + tag_table['Tag'][i] for i in range(len(tag_table))])
    movie.columns = movie_col
    movie = movie.drop('tag', axis=1)
    return movie


def feature_extraction(data):
    # actions: we use top 50 movies as our actions for recommendations
    actions = data.groupby('movie_id').size().sort_values(ascending=False)[:50]
    actions = list(actions.index)

    # user_feature: tags they've watched for non-top-50 movies normalized per user
    user_feature = data[~data['movie_id'].isin(actions)]
    user_feature = user_feature.drop(['movie_id', 'rating', 'timestamp', 'movie_name'], axis=1)
    user_feature = user_feature.groupby('user_id').aggregate('sum')
    user_feature = user_feature.div(user_feature.sum(axis=1), axis=0)

    # streaming_batch: the result for testing bandit algrorithms
    top50_data = data[data['movie_id'].isin(actions)]
    top50_data = top50_data.sort_values('timestamp', ascending=True)
    streaming_batch = top50_data['user_id']

    # reward_list: if rating >=3, the user will watch the movie
    top50_data['reward'] = np.where(top50_data['rating'] >= 3, 1, 0)
    reward_list = top50_data[['user_id', 'movie_id', 'reward']]
    reward_list = reward_list[reward_list['reward'] == 1]
    return streaming_batch, user_feature, actions, reward_list


def main():
    # read and preprocess the movie data
    movie = pd.read_table(join(RAW_DATA_DIR, 'movies.dat'), sep='::', names=['movie_id', 'movie_name', 'tag'], engine='python', encoding='latin-1')
    movie = movie_preprocessing(movie)

    # read the ratings data and merge it with movie data
    rating = pd.read_table(join(RAW_DATA_DIR, "ratings.dat"), sep="::",
                           names=["user_id", "movie_id", "rating", "timestamp"], engine='python', encoding='latin-1')
    data = pd.merge(rating, movie, on="movie_id")

    # extract feature from our data set
    streaming_batch, user_feature, actions, reward_list = feature_extraction(data)

    # dump to csv
    streaming_batch.to_csv(join(PROCESSED_DATA_DIR, "streaming_batch.csv"), sep='\t', index=False)
    user_feature.to_csv(join(PROCESSED_DATA_DIR, "user_feature.csv"), sep='\t')

    pd.DataFrame(actions, columns=['movie_id']).to_csv(join(PROCESSED_DATA_DIR, "actions.csv"), sep='\t', index=False)
    reward_list.to_csv(join(PROCESSED_DATA_DIR, "reward_list.csv"), sep='\t', index=False)

    action_context = movie[movie['movie_id'].isin(actions)]
    action_context.to_csv(join(PROCESSED_DATA_DIR, "action_context.csv"), sep='\t', index = False)

if __name__ == '__main__':
    main()
