import glob
import os
import pickle
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams



with open('capstone_stopwords','r') as f:
    stopwords_ = set(word.strip('\n') for word in f)
punctuation_ = set(string.punctuation)
path = '/Users/ale/Dropbox (Yadlin Family)/galvanize/capstone/*.txt'

with open("gene_dictionary_lower.pickle", "rb") as dict_gene:
        gene_dict_lower = pickle.load(dict_gene)

gene_keys_lower = set(gene_dict_lower.keys())
gene_values_lower = set(gene_dict_lower.values())

with open("drug_dictionary_lower.pickle", "rb") as dict_drug:
        drug_dict_lower = pickle.load(dict_drug)

drug_keys_lower = set(drug_dict_lower.keys())
drug_values_lower = set(drug_dict_lower.values())

with open("greek_alphabet.pickle", "rb") as dict_greek:
        greek_alphabet_dict = pickle.load(dict_greek)

greek_keys = set(greek_alphabet_dict.keys())
greek_values = set(greek_alphabet_dict.values())

def read_file(r_filename)
    '''read file and remove nuisances'''
    with open(r_filename),'rb') as r_file:
        txt = (r_file.read().decode('latin-1').replace('/',' ')
                 .replace('\n',' ').replace('\'', '').replace('"',"").replace("'",''))
    return txt

def read_files(files):
    '''read files from list of file names (path included)'''
    file_dict = {}
    for filepath in files:
        path, file_name = os.path.split(filepath)
        file_dict[file_name] = read_file(filepath)
    return file_dict

def filter_tokens(sent):
    ''' remove stopwords and punctuation'''
    return([w for w in sent if not w in stopwords_ and not w in punctuation_])


def dict_replace(sentence):  #(text)
    '''replace alternative gene and drug names for main names'''
    #for sentence in text:
    for idx, term in enumerate(sentence):
        if term in gene_keys_lower:
            sentence[idx] = gene_dict_lower[term]
        if term in drug_keys_lower:
            sentence[idx] = drug_dict_lower[term]
        if term in greek_keys:
            sentence[idx] = greek_dict[term]
    return sentence #text

def join_sent_ngrams(input_tokens, n):
    '''make n-grams containing the word not'''
    # first add the 1-gram tokens
    ret_list = list(input_tokens)

    #then for each n
    for i in range(2,n+1):
        #print(ret_list)
        # add each n-grams to the list
        ret_list.extend(['-'.join(tgram) for tgram in ngrams(input_tokens, i) if 'not' in tgram])

    return(ret_list)

def sentence_extractor(txt):
    '''extract sentences from text'''
    return sent_tokenize(txt)

def split_hyphen_word(word):
    ''' split hypehnated words unless second word starts with digit '''
    if '-' not in word:
        return [word]
    if len(word)-1 > word.index('-'):
        if not((word[word.index('-')+1]).isdigit()):
            return word.split('-')
        else:
            return [word]
    else:
        return [word[:-1]]

def split_hyphen_words(sent):
    ''' given a sentence with hyphenated words return unhephenyted sentence'''
    output = []
    for word in sent:
        output.extend(split_hyphen_word(word))
    return output

def is_valid_word(word):
    ''' check if word is a valid token '''
    if len(word)<3:
        return False
    exclude = set(string.digits + ".+*#,-")
    for character in word:
        if character not in exclude:
            return True
    return False

def tokenize(txt):
    '''tokenize txt -->sentneces'''
    txt=txt.lower()
    tokens = word_tokenize(txt)
    tokens_filtered = filter_tokens(tokens)
    tokens_filtered = [x.replace(x,x[:-1]) if x[-1]=='-' or x[-1]=='+' else x for x in tokens_filtered]
    tokens_filtered = dict_replace(tokens_filtered)
    stemmer_porter = PorterStemmer()
    tokens_stemporter = [stemmer_porter.stem(token) for token in tokens_filtered]
    tokens_stemporter = [x.replace('relaps','resist') for x in tokens_stemporter]
    tokens_ngrams = join_sent_ngrams(tokens_stemporter, 2)
    tokens_ngrams = [x.replace('not-sensit','resist') for x in tokens_ngrams]
    tokens_ngrams = [x.replace('not-resist','sensit') for x in tokens_ngrams]
    tokens_ngrams = [x.replace('egfr1','egfr') for x in tokens_ngrams]
    tokens_ngrams = split_hyphen_words(tokens_ngrams)
    return [word for word in tokens_ngrams if is_valid_word(word)]

def tokenize_doc(doc_name,doc_txt):
    '''tokenize all sentences in doc'''
    sentences = sent_tokenize(doc_txt)
    for sent in sentences:
        #print(sent)
        yield (doc_name, sent)

def tokenize_many_docs(file_path):
    '''tokenize all docs in filepath'''
    files = glob.glob(path)
    file_dict = read_files(files)
    for file_name, file_txt in file_dict.items():
        yield from tokenize_doc(file_name,file_txt)
