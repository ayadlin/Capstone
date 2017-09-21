import tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import pickle
from itertools import compress



path = '/Users/ale/Dropbox (Yadlin Family)/galvanize/capstone/*.txt'
path_aws = '/home/ubuntu/Capstone/data/txt/*.txt'

def create_data_frame(path=path_aws,
                      doc_reader=tokenizer.tokenize_many_docs,
                      tokenizer=tokenizer.tokenize,
                      short_list=False,
                      min_df=0.0000001):
    output = list(doc_reader(path))
    doc_names, orig_sentences = zip(*output)
    corpus = orig_sentences
    tf = CountVectorizer(tokenizer=lambda x: tokenizer(x, short_list=short_list), min_df=min_df)
    document_tf_matrix =tf.fit_transform(corpus).todense()
    papers_df = pd.DataFrame(document_tf_matrix)
    papers_df.columns = tf.get_feature_names()
    papers_df.sort_index(axis=1, inplace=True)
    non_empty_rows = list(compress(range(len(papers_df.any(axis=1))), papers_df.any(axis=1)))
    papers_df = papers_df[papers_df.any(axis=1)]
    sentences = [orig_sentences[i] for i in non_empty_rows]
    files = [doc_names[i] for i in non_empty_rows]
    papers_df.insert(0,'sentences',sentences)
    papers_df.insert(0,'files',files)
    #papers_df = papers_df[papers_df[tf.get_feature_names()].any(axis=1)].reset_index().drop('index',axis=1)
    return papers_df


def sparse_create_data_frame(path=path_aws,
                      doc_reader=tokenizer.tokenize_many_docs,
                      tokenizer=tokenizer.tokenize,
                      short_list=False,
                      min_df=0.005):
    output = list(doc_reader(path))
    doc_names, orig_sentences = zip(*output)
    corpus = orig_sentences
    tf = CountVectorizer(tokenizer=lambda x: tokenizer(x, short_list=short_list), min_df=min_df)
    vocab_matrix =tf.fit_transform(corpus)
    vocab_columns = tf.get_feature_names()
    return [vocab_matrix, vocab_columns, doc_names, orig_sentences]


def create_corpus(path=path, doc_reader=tokenizer.tokenize_many_docs):
    output = list(doc_reader(path))
    doc_names, orig_sentences = zip(*output)
    return list(doc_names), list(orig_sentences)


def process_data(corpus,
                 tokenizer=tokenizer.tokenize,
                 min_df=0.005,
                 short_list=False,
                 mat_out='vocab_matrix',
                 vocab_out='vocabulary'):
    corpus =corpus
    tf = CountVectorizer(tokenizer=lambda x: tokenizer(x, short_list=short_list), min_df=min_df)
    vocab_matrix =tf.fit_transform(corpus)
    vocabulary = tf.get_feature_names()
    with open(mat_out+'.pickle', 'wb') as handle:
        pickle.dump(vocab_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(vocab_out+'.pickle', 'wb') as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data has been processed and saved')
    return [vocab_matrix, vocabulary]

def open_pickle(filepath):
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def write_pickle(filepath, var):
    with open(filepath, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pandify_data(sparse_mat,vocabulary):
    if isinstance(sparse_mat, str):
        sparse_mat = open_pickle(sparse_mat)
    if isinstance(vocabulary, str):
        sparse_mat = open_pickle(vocabulary)
    df = pd.DataFrame(sparse_mat)
    df.columns = vocabulary
    df.sort_index(axis=1, inplace=True)
    return df

def add_files_and_text(df, docs, sent):
    non_empty_rows = list(compress(range(len(df.any(axis=1))), df.any(axis=1)))
    df = df[df.any(axis=1)]
    sentences = [sent[i] for i in non_empty_rows]
    files = [docs[i] for i in non_empty_rows]
    df.insert(0,'sentences',sentences)
    df.insert(0,'files',files)
    return df

def from_papers_to_panda(path=path_aws,
                      doc_reader=tokenizer.tokenize_many_docs,
                      tokenizer=tokenizer.tokenize,
                      short_list=False,
                      min_df=0.0000001,
                      mat_out='vocab_matrix',
                      vocab_out='vocabulary'):
    doc_names, orig_sentences = create_corpus(path=path,
                                              doc_reader=doc_reader)
    processed_data = process_data(corpus=orig_sentences,
                                  tokenizer=tokenizer,
                                  min_df=min_df,
                                  short_list=short_list,
                                  mat_out=mat_out,
                                  vocab_out=vocab_out)
    df = pandify_data(processed_data[0].todense(), processed_data[1])
    df = add_files_and_text(df, doc_names, orig_sentences)
    return df
