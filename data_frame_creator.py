import tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np



path = '/Users/ale/Dropbox (Yadlin Family)/galvanize/capstone/*.txt'
path_aws = '/home/ubuntu/Capstone/data/txt/*.txt'

def create_data_frame(path=path_aws,
                      doc_reader=tokenizer.tokenize_many_docs,
                      tokenizer=tokenizer.tokenize,
                      min_df=0.005):
    output = list(doc_reader(path))
    doc_names, orig_sentences = zip(*output)
    corpus = orig_sentences
    tf = CountVectorizer(tokenizer=tokenizer, min_df=min_df)
    document_tf_matrix =tf.fit_transform(corpus).todense()
    papers_df = pd.DataFrame(document_tf_matrix)
    papers_df.columns = tf.get_feature_names()
    papers_df.sort_index(axis=1, inplace=True)
    papers_df.insert(0,'sentences',orig_sentences)
    papers_df.insert(0,'files',doc_names)
    papers_df = papers_df[papers_df[tf.get_feature_names()].any(axis=1)].reset_index().drop('index',axis=1)

    return papers_df

def sparse_create_data_frame(path=path_aws,
                      doc_reader=tokenizer.tokenize_many_docs,
                      tokenizer=tokenizer.tokenize,
                      min_df=0.005):
    output = list(doc_reader(path))
    doc_names, orig_sentences = zip(*output)
    corpus = orig_sentences
    tf = CountVectorizer(tokenizer=tokenizer, min_df=min_df)
    vocab_matrix =tf.fit_transform(corpus)
    #papers_df = pd.DataFrame(document_tf_matrix)
    vocab_columns = tf.get_feature_names()
    #papers_df.sort_index(axis=1, inplace=True)
    #papers_df.insert(0,'sentences',orig_sentences)
    #papers_df.insert(0,'files',doc_names)
    #papers_df = papers_df[papers_df[tf.get_feature_names()].any(axis=1)].reset_index().drop('index',axis=1)

    return [vocab_matrix, vocab_columns, doc_names, orig_sentences]
