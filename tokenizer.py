import glob
import os
import pickle
import string
import multiprocessing
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams



with open('capstone_stopwords.pickle','r') as f:
    stopwords_ = set(word.strip('\n') for word in f)
punctuation_ = set(string.punctuation)
#path = '/Users/ale/Dropbox (Yadlin Family)/galvanize/capstone/*.txt'

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

def union(a, b):
    ''' return the union of two lists '''
    return set(list(set(a) | set(b)))

def intersect(a, b):
    ''' return the intersection of two lists '''
    return set(list(set(a) & set(b)))

def read_file(r_filename):
    '''read file and remove nuisances'''
    with open(r_filename,'rb') as r_file:
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
        if term in gene_keys:
            sentence[idx] = gene_dict[term]
        if term in drug_keys:
            sentence[idx] = drug_dict[term]
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

def in_short_list(word):
    ''' check if word in short list'''
    short_list = ['sensit', 'resist']
    if word[0]=='#' and word[-1]=='#':
        return True
    if word in short_list:
        return True
    return False

def tokenize(txt, short_list=False):
    '''tokenize txt -->sentneces'''
    txt=txt.lower()
    tokens = word_tokenize(txt)
    tokens_filtered = filter_tokens(tokens)
    tokens_filtered = [x.replace(x,x[:-1]) if x[-1]=='-' or x[-1]=='+'
                        else x for x in tokens_filtered]
    tokens_filtered = split_hyphen_words(tokens_filtered)
    tokens_filtered = dict_replace(tokens_filtered)
    tokens_filtered = [token for token in tokens_filtered if len(token)>0]
    stemmer_porter = PorterStemmer()
    tokens_stemporter = [stemmer_porter.stem(token) if not token[0]=='#'
                          and not token[-1]=='#' else token
                          for token in tokens_filtered]
    tokens_stemporter = [x.replace('relaps','resist') for x in tokens_stemporter]
    tokens_ngrams = join_sent_ngrams(tokens_stemporter, 2)
    tokens_ngrams = [x.replace('not-sensit','resist') for x in tokens_ngrams]
    tokens_ngrams = [x.replace('not-resist','sensit') for x in tokens_ngrams]
    tokens_ngrams = [x.replace('egfr1','egfr') for x in tokens_ngrams]
    if short_list:
        #tokens_ngrams = [word for word in tokens_ngrams if is_valid_word(word)]
        return [word for word in tokens_ngrams if in_short_list(word)]
    return [word for word in tokens_ngrams if is_valid_word(word)]

def tokenize_doc(doc_name,doc_txt):
    '''tokenize all sentences in doc'''
    sentences = sent_tokenize(doc_txt)
    for sent in sentences:
        #print(sent)
        yield (doc_name, sent)

def tokenize_many_docs(file_path):
    '''tokenize all docs in filepath'''
    files = glob.glob(file_path)
    assert len(files)>0
    file_dict = read_files(files)
    assert len(file_dict)>0
    for file_name, file_txt in file_dict.items():
        yield from tokenize_doc(file_name,file_txt)

def parallel_tokenize_many_docs(file_path):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(tokenize_doc, file_path)
    return union(results,results)
