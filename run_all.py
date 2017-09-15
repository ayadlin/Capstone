import tokenizer
import pickle
import pandas as pd
import numpy as np
import glob
import os
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
import json
import sparse_matrix_functions
import scipy.sparse as scs
from IPython.display import display
import matplotlib.pyplot as plt
#%matplotlib inline

import data_frame_creator
import sparse_matrix_functions

s_path='Final_Run_Results/'
#def read_data():
#READ/SAVE DATA#
################################################################################
final_read = data_frame_creator.sparse_create_data_frame(short_list =True,min_df=0)

with open(s_path+'final_vocab_matrix.pickle', 'wb') as handle:
    pickle.dump(final_read[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'final_vocabulary.pickle', 'wb') as handle:
    pickle.dump(final_read[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'final_doc_list.pickle', 'wb') as handle:
    pickle.dump(final_read[2], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'final_sentence_list.pickle', 'wb') as handle:
    pickle.dump(final_read[3], handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'final_data.pickle', 'wb') as handle:
    pickle.dump(final_read, handle, protocol=pickle.HIGHEST_PROTOCOL)
################################################################################
#return None

#GET DATA PROCESSED AND READY
################################################################################


#PLACE DATA INTO VARIABLES
final_vocab_matrix=final_read[0]
final_vocab_matrix[final_vocab_matrix>1]=1

with open(s_path+'final_vocab_matrix_ones.pickle', 'wb') as handle:
    pickle.dump(final_vocab_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

final_vocabulary = final_read[1]
final_doc_list=final_read[2]
final_sentence_list=final_read[3]

#MAKE VOCABULARY INDEX -->WHIHC COLUMN IS EACH GENE OR DRUG
vocab_idx=sparse_matrix_functions.column_indexing(final_vocabulary)

with open(s_path+'final_vocabulary_idx.pickle', 'wb') as handle:
    pickle.dump(vocab_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

#MATRIX INCLUDES ANY DRUG/GENE INTERACTIONS: DONT CARE SENSITIVITY OR RESISTANCE
vocab_matrix_all = sparse_matrix_functions.pick_network_type(final_vocab_matrix, final_vocabulary,kind='a')

with open(s_path+'any_vocab_matrix.pickle', 'wb') as handle:
    pickle.dump(vocab_matrix_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

#MATRIX INCLUDES ONLY DRUG/GENE INTERACTIONS OF RESISTANCE
vocab_matrix_resist = sparse_matrix_functions.pick_network_type(final_vocab_matrix, final_vocabulary,kind='r')

with open(s_path+'resist_vocab_matrix.pickle', 'wb') as handle:
    pickle.dump(vocab_matrix_resist, handle, protocol=pickle.HIGHEST_PROTOCOL)

#MATRIX INCLUDES ONLY DRUG/GENE INTERACTIONS OF SENSISTIVITY
vocab_matrix_sensit = sparse_matrix_functions.pick_network_type(final_vocab_matrix, final_vocabulary,kind='s')

with open(s_path+'sensit_vocab_matrix.pickle', 'wb') as handle:
    pickle.dump(vocab_matrix_sensit, handle, protocol=pickle.HIGHEST_PROTOCOL)

#GET GENES IN NETWORK
network_genes=sparse_matrix_functions.get_network_rows(final_vocabulary)

with open(s_path+'network_genes.pickle', 'wb') as handle:
    pickle.dump(network_genes, handle, protocol=pickle.HIGHEST_PROTOCOL)

inverse_network_drugs = dict((v, k) for k, v in network_drugs.items())

with open(s_path+'inverse_network_genes.pickle', 'wb') as handle:
    pickle.dump(Inverse_network_genes, handle, protocol=pickle.HIGHEST_PROTOCOL)

#GET DRUGS IN NETWORK
network_drugs =sparse_matrix_functions.get_network_columns(final_vocabulary)

with open(s_path+'network_drugs.pickle', 'wb') as handle:
    pickle.dump(network_drugs, handle, protocol=pickle.HIGHEST_PROTOCOL)

inverse_network_drugs = dict((v, k) for k, v in network_drugs.items())

with open(s_path+'inverse_network_drugs.pickle', 'wb') as handle:
    pickle.dump(inverse_network_drugs, handle, protocol=pickle.HIGHEST_PROTOCOL)
#GENERATE NETWORK MATRICES
################################################################################

# CREATE GENERAL INTERACTION NETWORK MATRIX

all_network_matrix = sparse_matrix_functions.make_network_matrix(vocab_matrix_all, final_vocabulary)

with open(s_path+'any_network_matrix.pickle', 'wb') as handle:
    pickle.dump(all_network_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

#CREATE RESISTANT NETWORK MATRIX
resist_network_matrix = sparse_matrix_functions.make_network_matrix(vocab_matrix_resist, final_vocabulary)

with open(s_path+'resist_network_matrix.pickle', 'wb') as handle:
    pickle.dump(resist_network_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

#CREATE SENSITIVITY NETWORK MATRIX
sensit_network_matrix = sparse_matrix_functions.make_network_matrix(vocab_matrix_sensit, final_vocabulary)

with open(s_path+'sensit_network_matrix.pickle', 'wb') as handle:
    pickle.dump(sensit_network_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)


#EXTRACT NETWORKS INFORMATION (PAIRS AND INTERACTION NUMBER)
################################################################################


pairs_a, counts_a = sparse_matrix_functions.extract_gene_drug_pairs(all_network_matrix, network_genes, network_drugs)

with open(s_path+'pairs_any.pickle', 'wb') as handle:
    pickle.dump(pairs_a, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'counts_any.pickle', 'wb') as handle:
     pickle.dump(counts_a, handle, protocol=pickle.HIGHEST_PROTOCOL)

pairs_r, counts_r = sparse_matrix_functions.extract_gene_drug_pairs(resist_network_matrix, network_genes, network_drugs)

with open(s_path+'pairs_resist.pickle', 'wb') as handle:
    pickle.dump(pairs_r, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'counts_resist.pickle', 'wb') as handle:
     pickle.dump(counts_r, handle, protocol=pickle.HIGHEST_PROTOCOL)



pairs_s, counts_s = sparse_matrix_functions.extract_gene_drug_pairs(sensit_network_matrix, network_genes, network_drugs)

with open(s_path+'pairs_sensit.pickle', 'wb') as handle:
    pickle.dump(pairs_s, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(s_path+'counts_sensit.pickle', 'wb') as handle:
     pickle.dump(counts_s, handle, protocol=pickle.HIGHEST_PROTOCOL)


#DICTIONARY BACK TO ORIGINAL DATA COLUMNS FOR INTERACTING GENE AND DRUG PAIRS
###############################################################################

original_indeces_a = sparse_matrix_functions.back_to_original_indeces(pairs_a, vocab_idx)
with open(s_path+'original_indices_any.pickle', 'wb') as handle:
    pickle.dump(original_indeces_a, handle, protocol=pickle.HIGHEST_PROTOCOL)

original_indeces_r = sparse_matrix_functions.back_to_original_indeces(pairs_r, vocab_idx)

with open(s_path+'original_indices_resist.pickle', 'wb') as handle:
    pickle.dump(original_indeces_r, handle, protocol=pickle.HIGHEST_PROTOCOL)

original_indeces_s = sparse_matrix_functions.back_to_original_indeces(pairs_s, vocab_idx)

with open(s_path+'original_indices_sensit.pickle', 'wb') as handle:
    pickle.dump(original_indeces_s, handle, protocol=pickle.HIGHEST_PROTOCOL)


#TEST
################################################################################
sensit_evidence_akt_lapatinib = sparse_matrix_functions.get_evidence_sentences(
'akt','lapatinib','s',7,final_read, original_indeces_s)

sensit_evidence_akt_lapatinib
