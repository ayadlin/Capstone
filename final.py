import tokenizer
import data_frame_creator
import sparse_matrix_functions
#import graph_maker
import make_NMF_model as NMF

#def get_evidence_sentences(gene, drug, r_s, max_number,data,original_indeces):
data  = data_frame_creator.open_pickle('final_data.pickle')


def find_evidence():
    gene = input('Which gene are you interested in?: ')
    drug = input('Interaction with which drug are you interested in?: ')
    max_number = input('How many evidence sentences, if available, would you like to see?: ')
    max_number =int(max_number) 
    r_s = input('For resistance intearctions press "r".\n'
                'For sensitivity interactions press "s".\n'
                'For general studies press "g".')
    if r_s == 'r' or r_s =='R':
        original_indeces = data_frame_creator.open_pickle('original_indices_resist.pickle')
    elif r_s == 's' or r_s == 'S':
        original_indeces = data_frame_creator.open_pickle('original_indices_sensit.pickle')
    else:
        original_indeces = data_frame_creator.open_pickle('original_indices_any.pickle')
    evidence = sparse_matrix_functions.get_evidence_sentences(gene, drug, r_s, max_number, data,original_indeces)
    #print(original_indeces)

    return evidence
