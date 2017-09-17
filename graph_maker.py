import tokenizer
import pickle
import data_frame_creator
import sparse_matrix_functions
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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx as nx
# %matplotlib inline

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

resist_network_matrix = data_frame_creator.open_pickle('resist_network_matrix.pickle')
sensit_network_matrix = data_frame_creator.open_pickle('sensit_network_matrix.pickle')
all_network_matrix = data_frame_creator.open_pickle('any_network_matrix.pickle')

dg = data_frame_creator.open_pickle('gene_or_drug.pickle')

def process_genes():
    genes = input('For what genes would you like to get drug interaction information?:, enter "all" for full network  ')
    if genes == 'all':
        processed_gene_list=list(network_genes.values())
        #processed_gene_list = ['a']
    else:
        gene_list=genes.split(', ')
        processed_gene_list = []
        for id in gene_list:
            try:
                gene_dict[id]
                processed_gene_list.append(gene_dict[id])
            except KeyError:
                print("Gene: {} not in list".format(id))
    return processed_gene_list



def extract_gene_drug_interaction(X,genes_list):

    interaction_dict={}
    genes =[]
    for gene in genes_list:
        multiplier = np.zeros((X.shape[0],1))
        idx = inverse_network_genes[gene]
        multiplier[idx]=1
        value = X.T.dot(multiplier)
        drug_idx = value.nonzero()[0]
        if sum(value) == 0:
            print('\n no drug interactions with gene {} were found in our database'.format(gene))
        else:
            #interaction_dict[gene] = value
            drug_idx = value.nonzero()[0]
            for idx in drug_idx:
                interaction_dict[gene, network_drugs[idx]]=value[idx,0]
            genes.append(gene)
    print('\nthese are the genes for which interactions with drugs were found:',genes )
    return interaction_dict


def get_user_input():
    genes_list = process_genes()
    kind = input('if you are interested on drug resistance evidence press "r"'
                'if you are interested on drug resistance evidence press "s"'
                 'if you are interested on general interactions press "g" '
                'if you are interested on all of above interactions press "a" ')
    interaction_list =[]
    if kind == 'r' or kind =='R':
        #if gene_list = ['a']:

        X = resist_network_matrix
        resistant_dict = extract_gene_drug_interaction(X,genes_list)
        interaction_list.append(resistant_dict)
    elif kind == 's' or kind =='S':
        X = sensit_network_matrix
        sensitive_dict = extract_gene_drug_interaction(X,genes_list)
        interaction_list.append(sensitive_dict)
    elif kind == 'g' or kind =='G':
        X = all_network_matrix
        sensitive_dict = extract_gene_drug_interaction(X,genes_list)
        interaction_list.append(sensitive_dict)
    else:
        X = all_network_matrix
        interaction_dict = extract_gene_drug_interaction(X,genes_list)
        interaction_list.append(interaction_dict)
        X = resist_network_matrix
        resistant_dict = extract_gene_drug_interaction(X,genes_list)
        interaction_list.append(resistant_dict)
        X = sensit_network_matrix
        sensitive_dict = extract_gene_drug_interaction(X,genes_list)
        interaction_list.append(sensitive_dict)
    return interaction_list


def gene_or_drug(vocabulary):
    gene_or_drug={}
    for item in vocabulary:
        if item in gene_values:
            gene_or_drug[item]='gene'
        if item in drug_values:
            gene_or_drug[item]='drug'
    return gene_or_drug


def draw_graph(G, gene_nodes, drug_nodes, weights, style='solid'):
    pos=nx.circular_layout(G) # positions for all nodes
    plt.figure(1,figsize=(12,12))
    nx.draw_networkx_nodes(G,pos,nodelist=gene_nodes,node_shape='s', node_color = 'gold', node_size = 2000)
    nx.draw_networkx_nodes(G,pos,nodelist=drug_nodes,node_shape='o', node_color = 'deepskyblue', node_size = 1000)
        #nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=weights, width=5.0, edge_cmap=plt.cm.coolwarm, style = style)
        #plt.savefig('edges.png')
        # labels
    nx.draw_networkx_labels(G,pos,font_size=10,font_family='arial')
    plt.axis('off')
    #plt.savefig("weighted_graph.png") # save as png
    #plt.show() # display
    return G

def get_node_set(edges):
    node_set=set()
    for node in edges():
        node_set.update(node)
    gene_nodes = []
    drug_nodes = []
    for item in list(node_set):
    #print(item)
        if dg['#'+item+'#']=='gene':
            gene_nodes.append(item)
        if dg['#'+item+'#']=='drug':
            drug_nodes.append(item)
    return [gene_nodes, drug_nodes]

def make_display_network(lst, path):


    if len(lst) ==1:
        G=nx.Graph()
        for (gene,drug), weight in lst[0].items():
            G.add_edge(gene[1:-1],drug[1:-1],weight=weight)
        weights = list(np.array(list(lst[0].values())))
        gene_nodes = get_node_set(G.edges)[0]
        drug_nodes = get_node_set(G.edges)[1]
        plt.figure(1,figsize=(12,12))
        G = draw_graph(G, gene_nodes, drug_nodes, weights, style='solid')
        plt.axis('off')
        plt.savefig("weighted_graph.png") # save as png
        plt.show() # display
        Graphs = G


    if len(lst) == 3:
        G1=nx.Graph()
        G2=nx.Graph()
        G3=nx.Graph()
        for (gene,drug), weight in lst[0].items():
            G1.add_edge(gene[1:-1],drug[1:-1],weight=weight)
        weights_1 = list(np.array(list(lst[0].values())))
        gene_nodes_1 = get_node_set(G1.edges)[0]
        drug_nodes_1 = get_node_set(G1.edges)[1]
        for (gene,drug), weight in lst[1].items():
            G2.add_edge(gene[1:-1],drug[1:-1],weight=weight)
        weights_2 = list(np.array(list(lst[1].values())))
        gene_nodes_2 = get_node_set(G2.edges)[0]
        drug_nodes_2 = get_node_set(G2.edges)[1]
        for (gene,drug), weight in lst[2].items():
            G3.add_edge(gene[1:-1],drug[1:-1],weight=weight)
        weights_3 = list(np.array(list(lst[2].values())))
        gene_nodes_3 = get_node_set(G3.edges)[0]
        drug_nodes_3 = get_node_set(G3.edges)[1]

        #for edge in list(G2.edges):
        #    if edge in list(G1.edges):
        #        G1.remove_edge(edge)
        #for edge in G3.edges:
        #    if edge in G1.edges:
        #        G1.remove_edge(edge)
        plt.figure(1,figsize=(12,12))

        G1 = draw_graph(G1, gene_nodes_1, drug_nodes_1, weights_1, style='dotted')
        G2 = draw_graph(G2, gene_nodes_2, drug_nodes_2, weights_2, style='dashed')
        G3 = draw_graph(G3, gene_nodes_3, drug_nodes_3, weights_3, style='solid')
        Graphs = [G1, G2, G3]

        plt.axis('off')
        plt.savefig(path+".png") # save as png
        #plt.show() # display

    return Graphs
