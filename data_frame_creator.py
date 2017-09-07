from tokenizer import tokenize
from tokenizer import tokenize_many_docs


path = '/Users/ale/Dropbox (Yadlin Family)/galvanize/capstone/*.txt'
output = list(tokenize_many_docs(path))
doc_names, orig_sentences = zip(*output)

corpus = orig_sentences
#print(corpus)
from sklearn.feature_extraction.text import CountVectorizer

tf = CountVectorizer(tokenizer=tokenize)#, min_df=0.01, vocabulary=vocab)#tokenizer=lambda doc: tokenize(str(doc)), lowercase=False)#vocabulary=vocab,tokenizer=lambda doc: doc, lowercase=False)

document_tf_matrix =tf.fit_transform(corpus).todense()

#print(sorted(tf.vocabulary_))

document_tf_matrix
#sorted(tf.vocabulary_.items(), key=lambda x: x[1])
