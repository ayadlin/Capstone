import tokenizer
import data_frame_creator
import sparse_matrix_functions
import graph_maker
import make_NMF_model as NMF
import pyspark.ml.recommendation

#def get_evidence_sentences(gene, drug, r_s, max_number,data,original_indeces):
data  = data_frame_creator.open_pickle('final_data.pickle')


def find_network():
    G = graph_maker.make_graph_interactive()
    return G

#def find_evidence():
#    evidence = provide_evidence(data)
#    return evidence

def drug_predictions():
    predict_dict = provide_drug_predictions()
    return predict_dict
