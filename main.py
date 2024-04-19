import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import torch

START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_corpus(tokenized_texts):
    """ 
        Params:
            tokenized_texts: pandas Series of lists of tokens
        Return:
            list of lists, with words from each of the processed texts
    """
    return [[START_TOKEN] + text + [END_TOKEN] for text in tokenized_texts]

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1

    # ------------------
    # Write your implementation here.
    all_corpus_words = [y for x in corpus for y in x]
    corpus_words = sorted(set(all_corpus_words))
    num_corpus_words = len(corpus_words)
    # end of implementation

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)):
                Co-occurence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    # ------------------
    # Write your implementation here.
    for i in range(len(words)):
        word2Ind[words[i]] = i
    M = np.zeros((num_words,num_words))
    for doc in corpus:
      for i in range(len(doc)):
        ind1 = word2Ind[doc[i]]
        start = i - window_size
        end = i + window_size + 1
        if i < window_size:
          start = 0
        if i > len(doc) - window_size - 1:
          end = len(doc)
        for j in range(start, end):
          if j == i:
            continue
          ind2 = word2Ind[doc[j]]
          M[ind1, ind2] += 1

    # end of implementation

    return M, word2Ind

def calculate_PPMI(m, kind, k=2):

   alpha = 0.75
   # print(m)
   sum_m = torch.sum(m)
   # print(sum_m)
   sum_m_weighted = torch.sum(torch.float_power(m,alpha))
   # print(sum_m_weighted)


   row_prob = {}
   if kind == 'weighted':
      
      col_prob_weighted = {}
      for i in range(m.shape[0]):
         row_prob[str(i)] = torch.sum(m[i,:])/sum_m
      for j in range(m.shape[1]):
         col_prob_weighted[str(j)] = torch.sum(torch.float_power(m[:,j],alpha))/sum_m_weighted
      for i in range(m.shape[0]):
         print(i)
         for j in range(m.shape[1]):
            if int(m[i,j])!=0 and row_prob[str(i)]!=0 and col_prob_weighted[str(j)]!=0:
               p_i = row_prob[str(i)]
               p_j = col_prob_weighted[str(j)]
               p_ij = m[i,j]/sum_m
               m[i, j] = max(torch.log2(p_ij/(p_i*p_j)),0)


   if kind == 'add_k':
      
      col_prob = {}
      m += k
      
      for i in range(m.shape[0]):
         row_prob[str(i)] = torch.sum(m[i,:])/sum_m
      for j in range(m.shape[1]):
         col_prob[str(j)] = torch.sum(m[:,j])/sum_m
      for i in range(m.shape[0]):
         print(i)
         
         for j in range(m.shape[1]):
            p_i = row_prob[str(i)]
            p_j = col_prob[str(j)]
            if m[i,j] != k:
               p_ij = m[i,j]/sum_m
            else:
               p_ij = k/sum_m
            m[i, j] = max(torch.log2(p_ij/(p_i*p_j)),0)

   return m   


def create_features_from_ppmi(tokens, m, word2Ind):
    print(tokens)
    feat = np.zeros(m.shape[1])
    for token in tokens:
        ind = word2Ind[token]
        feat = np.add(feat, m[ind,:])
    return feat


if __name__ == "__main__":
    data_ = pd.read_csv('processed_data.csv')
    data_['tokenized_message_final'] = data_['tokenized_message_final'].apply(lambda x: x[1:-1].split(','))

    corpus = read_corpus(data_['tokenized_message_final'])
    corpus_words, num_corpus_words = distinct_words(corpus)
    M, word2Ind = compute_co_occurrence_matrix(corpus, window_size=2)
    M = torch.tensor(M)
    # weighted_ppmi_matrix = calculate_PPMI(M, 'weighted')
    # add_2_ppmi_matrix = calculate_PPMI(M, 'add_k', k=2)
    # torch.save(weighted_ppmi_matrix, 'weighted_ppmi_matrix.pt')
    # torch.save(add_2_ppmi_matrix, 'add_2_ppmi_matrix.pt')

    weighted_ppmi_matrix = torch.load('weighted_ppmi_matrix.pt')
    features = data_['tokenized_message_final'].apply(lambda x: create_features_from_ppmi(x, M, word2Ind))

    features = torch.vstack(list(features))
    torch.save(features, 'features.pt')
    features = torch.load('features.pt')
    feat_np = features.numpy()
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0).fit(feat_np, data_['label (depression result)'])
    print(clf.score(feat_np, data_['label (depression result)']))






   

