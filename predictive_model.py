import graph_maker
import pickle
import pandas as pd
import numpy as np
import data_frame_creator

vocabulary = data_frame_creator.open_pickle('final_vocabulary.pickle')

inverse_network_genes = data_frame_creator.open_pickle('inverse_network_genes.pickle')
inverse_network_drugs = data_frame_creator.open_pickle('inverse_network_drugs.pickle')

gene_dict = data_frame_creator.open_pickle("gene_dictionary_final.pickle")

gene_keys = set(gene_dict.keys())
gene_values = set(gene_dict.values())

drug_dict = data_frame_creator.open_pickle("drug_dictionary_final.pickle")

drug_keys = set(drug_dict.keys())
drug_values = set(drug_dict.values())

network_genes=data_frame_creator.open_pickle('network_genes.pickle')
network_drugs =data_frame_creator.open_pickle('network_drugs.pickle')

gene_any_pd = data_frame_creator.open_pickle('gene_any_pd.pickle')
drug_any_pd = data_frame_creator.open_pickle('drug_any_pd.pickle')

gene_resist_pd = data_frame_creator.open_pickle('gene_resist_pd.pickle')
drug_resist_pd = data_frame_creator.open_pickle('drug_resist_pd.pickle')

gene_sensit_pd = data_frame_creator.open_pickle('gene_sensit_pd.pickle')
drug_sensit_pd = data_frame_creator.open_pickle('drug_sensit_pd.pickle')

def get_user_input():
    genes = input('For what genes would you like to get drug interaction information?:, enter "all" for full network  ')
    genes_list = graph_maker.process_genes(genes)
    gene_idx = []
    for gene in genes_list:
        idx = inverse_network_genes[gene]
        gene_idx.append(idx)
    kind = input('\nif you are interested on drug resistance evidence press "r"\n'
                'if you are interested on drug resistance evidence press "s"\n'
                 'if you are interested on general interactions press "g"\n'
                'if you are interested on all of above interactions press "a".\n'
                'Please eneter your choice: ')
    if kind == 'r' or kind == 'R':
        gene_factors = gene_resist_pd.loc[gene_idx,'features']
        drug_factors = drug_resist_pd.loc[:,'features']
        #model = pyspark.ml.recommendation.ALSModel.load('resist_NMF_model')
    if kind == 's' or kind == 'S':
        gene_factors = gene_sensit_pd.loc[gene_idx,'features']
        drug_factors = drug_sensit_pd.loc[:,'features']
    if kind == 'a' or kind == 'A':
        gene_factors = gene_any_pd.loc[gene_idx,'features']
        drug_factors = drug_any_pd.loc[:,'features']
    drug_number = input('How many drug names would you like to see? ' )
    order = input ('Would you like to display the drugs in ascending or descendicg order. a/d: ')
    if order == 'a' or order == 'A':
        asc = True
    else:
        asc = False
    return genes_list, kind, gene_factors, drug_factors,drug_number, asc

def provide_drug_predictions():
    genes_list, kind, gene_factors, drug_factors,drug_number, asc = get_user_input()
    predicted_drugs = {}
    for idx,row in enumerate(gene_factors):
        drug_list = []
        gene = network_genes[gene_factors.index[idx]]
        new_drugs = drug_factors.apply(lambda x:np.dot(x,row))[0:drug_number]
        print(new_drugs)
        if asc:
            new_drugs = new_drugs.sort_values(ascending=asc)[0:drug_number]
            print(new_drugs)
        #print(new_drugs)
        for idx, drug in enumerate(new_drugs):
           drug_list.append(network_drugs[new_drugs.index[idx]][1:-1])
        predicted_drugs[gene] = drug_list
    return predicted_drugs
