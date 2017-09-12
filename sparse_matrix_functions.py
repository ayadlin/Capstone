import numpy as np
import scipy.sparse as scs
from itertools import compress
import string
import pickle

with open("gene_dictionary_final.pickle", "rb") as dict_gene:
        gene_dict = pickle.load(dict_gene)

gene_keys = set(gene_dict.keys())
gene_values = set(gene_dict.values())

with open("drug_dictionary_final.pickle", "rb") as dict_drug:
        drug_dict = pickle.load(dict_drug)

drug_keys = set(drug_dict.keys())
drug_values = set(drug_dict.values())


with open("greek_alphabet.pickle", "rb") as dict_greek:
        greek_alphabet_dict = pickle.load(dict_greek)

greek_keys = set(greek_alphabet_dict.keys())
greek_values = set(greek_alphabet_dict.values())

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

def select_rows(X,kind='A'):
    if kind == 'r' or kind =='R':
        ind =vocab_matrix[:,vocab_columns.index('resist')]
        X=X[ind.nonzero()[0],:]
    elif kind == 's' or kind =='S':
        ind =vocab_matrix[:,vocab_columns.index('sensit')]
        X=X[ind.nonzero()[0],:]
    else:
        X==X
    return X

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

def extract_gene_drug_pairs(matrix, network_genes, network_drugs):
    a =zip(list(matrix.nonzero()[0]),list(matrix.nonzero()[1]))
    pairs = []
    counts = []
    for pair in a:
        gendrug = (network_genes[pair[0]], network_drugs[pair[1]])
        r =np.zeros((matrix.shape[0],1))
        r[pair[0]]=1
        c =np.zeros((matrix.shape[1],1))
        c[pair[1]]=1
        count =(r.T).dot(matrix.dot(c))
        pairs.append(gendrug)
        counts.append(count)
    return (pairs, counts)

# def back_to_original_index(pairs):
#     original_index = []
#     for item in pairs:
#         original = (col_idx[item[0]], col_idx[item[1]])
#         original_index.append(original)
#     return original_index

def back_to_original_index(pair, orig_dict):
    original = (orig_dict[pair[0]],orig_dict[pair[1]])
    return pair, original

def back_to_original_indeces(pairs, orig_dict):
    original_indeces = {}
    for item in pairs:
        original = back_to_original_index(item, orig_dict)
        original_indeces[original[0]]=original[1]
    return original_indeces

def get_evidence_sentences(gene, drug, r_s, max_number,data,original_indeces):
    vocab_matrix = data[0]

    doc_names = data[2]
    orig_sentences = data[3]
    gen = '#'+gene+'#'
    drg = '#'+drug+'#'
    gene = gene.lower()
    drug = drug.lower()
    if gen not in gene_values:
        try:
            gen=gene_dict[gene]
        except KeyError:
            return ("Gene: {} not in list".format(gene))
    if drg not in drug_values:
        try:
            drg=drug_dict[drug]
        except KeyError:
            return ("Drug: {} not in list".format(drug))
    try:
        indices = original_indeces[(gen, drg)]
    except KeyError:
        return('There is no evidence of interaction'
        ' between the gene {} and the drug {}'. format(gen[1:-1],drg[1:-1]))
    test_col = scs.lil_matrix((vocab_matrix.shape[1],1))
    #print(test_col.shape)
    test_col[list(indices)]=1
    if r_s == 'S' or r_s == 's':
        test_col[vocab_matrix.shape[1]-1]=1
    elif r_s == 'R'or r_s == 'r':
         test_col[vocab_matrix.shape[1]-2]=1
    else:
         print('Returning evidence of both sensitivity and resistant')
    index_evidence = (vocab_matrix*test_col>2).nonzero()[0]
    evidence_list = []
    #gene_key = [k for k, v in gene_dict.items() if v == gen]
    #drug_key = [k for k, v in drug_dict.items() if v == drg]
    while len(evidence_list) < max_number and len(evidence_list) < len(index_evidence) :
        for index in index_evidence:
            sent = orig_sentences[index]#.lower()
            #gene_evd = [word for word in sent.replace(',','').replace('.','').replace(':','').split(' ') if word in gene_key][0]
            #drug_evd = [word for word in sent.replace(',','').replace('.','').replace(':','').split(' ') if word in drug_key][0]
            if len(sent)<500:
                evidence_list.append('{}: {}'.format(doc_names[index] ,sent))
            else:
                evidence_list.append('{}: sentence too long to display'.format(doc_names[index]))
    return evidence_list
