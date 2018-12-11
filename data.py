import pandas as pd
import tensorflow as tf
from functools import partial
import numpy as np


reading_col_name = ['usr', 'prd', 'rating', 'content']
output_col_name = ['usr', 'prd', 'rating', 'content', 'doc_len', 'sen_len']
emb_col_name = ['wrd'] + [i for i in range(200)]


def build_dataset(filenames, embedding_filename, max_doc_len, max_sen_len, hierarchy):
    wrd_dict, wrd_index, embedding = load_embedding(embedding_filename)
    # read the data and transform them
    data_frames, usr_cnt, prd_cnt = read_files(filenames, wrd_index, max_doc_len,
                                               max_sen_len, hierarchy)
    print('usr_cnt: %d, prd_cnt: %d' % (usr_cnt, prd_cnt))

    datasets = []
    lengths = []
    # build the dataset
    for data_frame in data_frames:
        data = {col: data_frame[col].values for col in output_col_name}
        data['content'] = np.stack(data['content'])
        data['sen_len'] = np.stack(data['sen_len'])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        datasets.append(dataset)
        lengths.append(len(data_frame))

    return datasets, lengths, embedding.values, usr_cnt, prd_cnt, wrd_dict


# load an embedding file
def load_embedding(filename):
    data_frame = pd.read_csv(filename, sep=' ', header=0, names=emb_col_name)
    embedding = data_frame[emb_col_name[1:]]
    wrd_dict = data_frame['wrd'].tolist()
    wrd_index = {s: i for i, s in enumerate(wrd_dict)}
    return wrd_dict, wrd_index, embedding


# transform a sentence into indices
def sentence_transform(document, wrd_index, max_doc_len, max_sen_len, hierarchy=True):
    if hierarchy is True:
        sentence_index = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        # doc_len, sen_len = 0, np.zeros(max_doc_len)
        for i, sentence in enumerate(document):
            if i == max_doc_len:
                break
            j = 0
            for wrd in sentence:
                if j == max_sen_len:
                    break
                if wrd in wrd_index:
                    sentence_index[i][j] = wrd_index[wrd]
                    j += 1
            # doc_len = i + 1
            # sen_len[i] = j
    else:
        sentence_index = np.zeros((max_doc_len, ), dtype=np.int)
        i = 0
        for wrd in document:
            if i == max_doc_len:
                break
            if wrd in wrd_index:
                sentence_index[i] = wrd_index[wrd]
                i += 1
    return sentence_index


def sen_len_transform(sen_len, max_doc_len):
    new_sen_len = np.zeros(max_doc_len)
    for i, l in enumerate(sen_len):
        if i == max_doc_len:
            break
        new_sen_len[i] = l
    return new_sen_len


def split_paragraph(paragraph, hierarchy=True):
    if hierarchy:
        sentences = paragraph.split('<sssss>')
        for i, _ in enumerate(sentences):
            sentences[i] = sentences[i].split()
    else:
        sentences = paragraph.split()
    return sentences


def read_files(filenames, wrd_index, max_doc_len, max_sen_len, hierarchy):
    data_frames = [pd.read_csv(filename, sep='\t\t', names=reading_col_name, engine='python')
                   for filename in filenames]
    print('Data frame loaded.')

    # count contents' length
    for df in data_frames:
        df['content'] = df['content'].transform(partial(split_paragraph, hierarchy=hierarchy))
        df['rating'] = df['rating'] - 1
        # df['max_sen_len'] = df['sen_len'].transform(lambda sen_len: max(sen_len))

    total_data = pd.concat(data_frames)
    # max_doc_len = total_data['doc_len'].max()
    # max_sen_len = total_data['max_sen_len'].max()
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
        df['content'] = df['content'].transform(
            partial(sentence_transform, wrd_index=wrd_index, max_doc_len=max_doc_len,
                    max_sen_len=max_sen_len, hierarchy=hierarchy))
        if hierarchy:
            df['sen_len'] = df['content'].transform(lambda document: np.sum(document != -1, axis=1))
            # df['sen_len'] = df['sen_len'].transform(partial(sen_len_transform,
            #                                                 max_doc_len=max_doc_len))
            df['doc_len'] = df['sen_len'].transform(lambda sen_len: np.count_nonzero(sen_len))
        else:
            df['doc_len'] = df['content'].transform(lambda sen_len: np.count_nonzero(sen_len))
            df['sen_len'] = df['doc_len']
    print('Contents indexed.')

    return data_frames, len(usr), len(prd)
