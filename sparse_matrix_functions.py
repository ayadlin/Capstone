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
