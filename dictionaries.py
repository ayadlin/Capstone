import pandas as pd
from collections import defaultdict
import string
import pickle
import json

'''Upload gene and drug info'''
genes = pd.read_excel('genes.xlsx')
drugs = pd.read_excel('Antineoplastic_Agent.xlsx')

genes = genes.fillna('')
drugs = drugs.fillna('')
genes.drop('Entrez Gene ID',axis=1, inplace=True)
genes.drop('Entrez Gene ID(supplied by NCBI)', axis=1, inplace=True)

################################################################################

''' FIRST THE GENES DICTIONARY '''
''' make all genes strings'''
genes = genes.applymap(str)

''' Create gene dictionary data frame '''
dict_genes = pd.DataFrame()

'''Set dictionary of genes keys using all alternative denominations'''
dict_genes['key'] = (genes['Approved Name'].str.cat(genes['Previous Symbols'],', ')
                     .str.cat(genes['Synonyms'], ', ').str.cat(genes['Accession Numbers']))

'''Set dictionary of genes values using most accepted denominations'''
dict_genes['value'] = genes['Approved Symbol']

'''Using dict_genes dataframe create the gene dictionary: gene_dict'''

gene_dict =defaultdict(list)
for row in dict_genes.iterrows():
    name_list = row[1]['key'].replace('"',',').split(',')
    main_name = row[1]['value']
    for name in name_list:
        if 'withdrawn' not in main_name and name != "" and len(name)>3:
            if main_name not in gene_dict[name]:
                gene_dict[name.lstrip()].append(main_name)

''' find keys with more than more value'''
long_values = []
for key in gene_dict.keys():
    if len(gene_dict[key]) !=1:
        long_values.append(key)

''' remove keys with more than one value'''

for key in long_values:
    del gene_dict[key]

''' if by mistake there is more than one vlaue per key keep the first one'''
for key in gene_dict.keys():
    gene_dict[key] = gene_dict[key][0]

''' make a list of all the keys that are also a value'''
values = set(gene_dict.values())
key_in_values = []
for key in gene_dict.keys():
    if key in values:
        key_in_values.append(key)

''' delete keys that are also a value'''
for key in key_in_values:
    del gene_dict[key]

''' make a lower case gene dictionary'''
gene_dict_lower = {}
for key, value in gene_dict.items():
    gene_dict_lower[key.lower()]=value.lower()

gene_keys_lower = set(gene_dict_lower.keys())
gene_values_lower = set(gene_dict_lower.values())

################################################################################

''' SECOND THE DRUG DICTIONARY '''

'''Using drugs dataframe create the gene dictionary: drug_dict'''
drug_dict={}
for row in drugs.iterrows():
    name_list = row[1]['Synonyms'].split('|')
    main_name = row[1]['Preferred Name']
    for name in name_list:
        drug_dict[name]=str(main_name)

''' make a lower case gene dictionary '''
drug_dict_lower = {}
for key, value in drug_dict.items():
    drug_dict_lower[key.lower()]=value.lower()

drug_keys_lower = set(drug_dict_lower.keys())
drug_values_lower = set(drug_dict_lower.values())

################################################################################

''' THIRD GREEK CHARACTERS DICTIONARY '''

greek_alphabet_dict = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}

################################################################################

''' FOURTH EXPORT DICTIONARIES TO JSON AND PICKLE FILES'''

with open('gene_dictionary.json', 'w') as fp:
    json.dump(gene_dict, fp, sort_keys=True, indent=4)
with open('gene_dictionary.pickle', 'wb') as handle:
    pickle.dump(gene_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('gene_dictionary_lower.json', 'w') as fp:
    json.dump(gene_dict_lower, fp, sort_keys=True, indent=4)
with open('gene_dictionary_lower.pickle', 'wb') as handle:
    pickle.dump(gene_dict_lower, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('drug_dictionary.json', 'w') as fp:
    json.dump(drug_dict, fp, sort_keys=True, indent=4)
with open('drug_dictionary.pickle', 'wb') as handle:
    pickle.dump(drug_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('drug_dictionary_lower.json', 'w') as fp:
    json.dump(drug_dict_lower, fp, sort_keys=True, indent=4)
with open('drug_dictionary_lower.pickle', 'wb') as handle:
    pickle.dump(drug_dict_lower, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('greek_alphabet.json', 'w') as fp:
    json.dump(greek_alphabet_dict, fp, sort_keys=True, indent=4)
with open('greek_alphabet.pickle', 'wb') as handle:
    pickle.dump(greek_alphabet_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
