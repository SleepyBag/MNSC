import pandas as pd
import tensorflow as tf
from functools import partial
import numpy as np


reading_col_name = ['usr', 'prd', 'rating', 'content']
output_col_name = ['usr', 'prd', 'rating', 'content', 'len']
emb_col_name = ['wrd'] + [i for i in range(200)]


def build_dataset(filenames, embedding_filename):
    wrd_index, embedding = load_embedding(embedding_filename)
    # read the data and transform them
    data_frames, usr_cnt, prd_cnt = read_files(filenames, wrd_index)

    datasets = []
    lengths = []
    # build the dataset
    for data_frame in data_frames:
        data = {col: data_frame[col].values for col in output_col_name}
        data['content'] = np.stack(data['content'])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        datasets.append(dataset)
        lengths.append(len(data_frame))

    return datasets, lengths, embedding.values, usr_cnt, prd_cnt


# load an embedding file
def load_embedding(filename):
    data_frame = pd.read_csv(filename, sep=' ', header=0, names=emb_col_name)
    embedding = data_frame[emb_col_name[1:]]
    wrd = data_frame['wrd'].tolist()
    wrd_index = {s: i for i, s in enumerate(wrd)}
    return wrd_index, embedding


# transform a sentence into indices
def sentence_transform(sentence, wrd_index, max_len):
    sentence_index = np.zeros(max_len, dtype=np.int) - 1
    i = 0
    for wrd in sentence:
        if wrd in wrd_index:
            sentence_index[i] = wrd_index[wrd]
            i += 1
    return sentence_index 


def read_files(filenames, wrd_index):
    data_frames = [pd.read_csv(filename, sep='\t\t', names=reading_col_name, engine='python')
                   for filename in filenames]
    print('Data frame loaded.')

    # count contents' length
    for df in data_frames:
        df['content'] = df['content'].transform(lambda sentence: sentence.split())
        df['len'] = df['content'].transform(lambda sentence: len(sentence))

    total_data = pd.concat(data_frames)
    max_len = total_data['len'].max()
    print('Length counted')

    # transform users and products to indices
    usr = total_data['usr'].unique().tolist()
    usr = {name: index for index, name in enumerate(usr)}
    prd = total_data['prd'].unique().tolist()
    prd = {name: index for index, name in enumerate(prd)}
    for df in data_frames:
        df['usr'] = df['usr'].map(usr)
        df['prd'] = df['prd'].map(prd)
    print('Users and products indexed.')

    # transform contents into indices
    for df in data_frames:
        df['content'] = df['content'].transform(partial(sentence_transform,
                                                        wrd_index=wrd_index,
                                                        max_len=max_len))
    print('Contents indexed.')

    return data_frames, len(usr), len(prd)
