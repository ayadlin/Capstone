# Resistance and Sensitivity: genomic interactions of cancer drugs
Work for Galvanize capstone project

## Steps:
### 1. Read PDF files to txt (console based)

### 2. Create dictionaries:
- Gene dictionary: genes have multiple names - associate all names to standard name
- Drug dictionary: drugs have multiple names - associate all names to standard name

### 3. Tokenize txt files:
- divide text into sentences
- sentences to word lists
- lower case words
- filter tokens for stop words or punctuation
- break hyphenated words
- identify gene names and translate to standard name
- identify drug names and translate to standard name
- stem words

### 4. Read and process data:
- Create document matrix
  - Use count vectorizer - does word occur in sentence or not
- Create vocabulary
- Create list of academic papers where sentences are found
- Create list of evidence sentence
- Use document matrix to create gene/drug network interaction matrix
- Create ditionary to associate genes and drugs to their indices in different matrices

### 5. Functionality:
- Use gene/drug network matrix to:
  - Generate networks of gene/drug interactions
  - Parse literature for evidence of gene/drug network interactions
  - Predict novel gene/drug interactions



