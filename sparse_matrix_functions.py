import numpy as np
import scipy.sparse as scs
from itertools import compress

def non_zero_rows(M):
    'identify all indeces in sparse matrix with value different to 0'
    M = scs.csr_matrix(M)
    num_nonzeros = np.diff(M.indptr)
    return list(compress(range(len(num_nonzeros !=0)), num_nonzeros !=0))

def remove_zero_rows(M):
    M = scs.csr_matrix(M)
    num_nonzeros = np.diff(M.indptr)
    return M[num_nonzeros != 0]

def column_indexing(columns):
    column_index={}
    for idx,name in enumerate(columns):
        column_index[name]=idx
    return column_index

def drop_columns(X,index_to_drop):
    to_keep = list(set(range(X.shape[1]-1))-set(index_to_drop))
    new_X = X[:,to_keep]
    return new_X

def make_network_matrix(X, vocabulary):
    #drug_columns_name =[i for i in vocabulary if i in drug_values]
    drug_columns_index =[vocabulary.index(i) for i in vocabulary if i in drug_values]
    gene_columns_index =[vocabulary.index(i) for i in vocabulary if i in gene_values]
    #gene_columns_name =[i for i in vocabulary if i in gene_values]
    gene_mat=X[:,gene_columns_index]
    drug_mat=X[:,drug_columns_index]
    network_matrix = gene_mat.T*drug_mat
    return network_matrix

def get_network_rows(vocabulary):
    genes = {}
    genes_columns_names=[i for i in vocabulary if i in gene_values]
    for idx,name in enumerate(genes_columns_names):
        genes[idx]=name
    return genes

def get_network_columns(vocabulary):
    drugs = {}
    drugs_columns_names=[i for i in vocabulary if i in drug_values]
    for idx,name in enumerate(drugs_columns_names):
        drugs[idx]=name
    return drugs
