# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:29:55 2020

@author: funrr
"""

import os

from nltk import tokenize

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.decomposition import PCA

ME_DIR = os.path.dirname(os.path.realpath(__file__))

work_dir = ME_DIR
embedding_file = work_dir + '/embedding/glove.twitter.27B.200d.txt'
emoji_embedding_file = work_dir + '/embedding/emoji2vec.txt'

train_file = work_dir + '/OLIDv1.0/olid-training-v1.0.tsv'
test_label_file = work_dir + '/OLIDv1.0/labels-levela.csv'
test_data_file = work_dir + '/OLIDv1.0/testset-levela.tsv'
#TODO num_classes = 3
embedding_dim = 200



    
##done
def sentenceTransform(sentenceList):
    #gives a list of sentences, return a cleaned, tokenized sentenceList, and calculate the longest sentence length
    token_sentence_list = []
    tknzr = tokenize.TweetTokenizer()

    for sentence in sentenceList:
        sentence = sentence.lower()      #Tolowercase    
        sentence = tknzr.tokenize(sentence)   #tokenize 
        token_sentence_list.append(sentence)
               
    for sentence in token_sentence_list:
        for idx, word in enumerate(sentence):
            if word == '@user' :
                sentence[idx] = '<user>'
            if word == 'url':
                sentence[idx] = '<url>'
            if word.isdigit():
                sentence[idx] = '<number>'
            if word[0] == '#':
                sentence[idx] = '<hashtag>'
            

    
        

    return token_sentence_list


##done
def ReadTrainDataset(file, task = 'a'):
    #read data from file, return label and raw sentence 
    #file: training data file, task:'a', 'b', or 'c'
    data = pd.read_csv(file, sep='\t')
    if task == 'a':
        data.replace(['NOT','OFF'], [0, 1], inplace=True)
        label = data.iloc[:,2].to_numpy()
    elif task == 'b':   
        data.replace(['UNT','TIN'], [0, 1], inplace=True)
        data.drop(columns=['subtask_a', 'subtask_c'], inplace = True)
        data.dropna(inplace = True)
    elif task == 'c':   
        data.replace(['IND','GRP','OTH'], [0, 1, 2], inplace=True)
        data.drop(columns=['subtask_a', 'subtask_b'], inplace = True)
        data.dropna(inplace = True)
        
    label = data.iloc[:,2].to_numpy()
    tweet = data.iloc[:,1].tolist()    
    return label, tweet
#Done

def ReadTestDataset(test_data_file, test_label_file, task):
    #read data from file, return label and raw sentence 
    #test_data_file: test data file, test_label_file: test label file
    #task:'a', 'b', or 'c'
    data = pd.read_csv(test_data_file, sep='\t')
    label = pd.read_csv(test_label_file, sep=',',header=None, names=['id','label'])
    data = pd.merge(data, label, on='id')
    if task == 'a':
        data.replace(['NOT','OFF'], [0, 1], inplace=True)
    
    elif task == 'b':
        data.replace(['UNT','TIN'], [0, 1], inplace=True)

    elif task == 'c':
        data.replace(['IND','GRP','OTH'], [0, 1, 2], inplace=True)
               
    tweet = data.iloc[:,1].tolist() 
    label = data.iloc[:,2].to_numpy()
    return label, tweet





#Done
class OLIDData(Dataset):
   #pytorch SICK data class; used by DataLoader
   #data: either 'train','test'/ task: 'a', 'b', 'c'
   
   def __init__(self, train=True, task = 'a'):
       self.data_dir = work_dir+'/OLIDv1.0'
       self.train = train   
       
       if task == 'a':
           if self.train:
               file = self.data_dir + '/olid-training-v1.0.tsv'
               print(file)
               self.label, self.tweet = ReadTrainDataset(file, task='a')       
           elif not self.train:
               test_data_file = self.data_dir + '/testset-levela.tsv'
               test_label_file = self.data_dir + '/labels-levela.csv'
               self.label, self.tweet = ReadTestDataset(test_data_file, test_label_file, task ='a')
  
       elif task == 'b':
           if self.train:
               file = self.data_dir + '/olid-training-v1.0.tsv'
               self.label, self.tweet = ReadTrainDataset(file, task='b')       
           elif not self.train:
               test_data_file = self.data_dir + '/testset-levelb.tsv'
               test_label_file = self.data_dir + '/labels-levelb.csv'
               self.label, self.tweet = ReadTestDataset(test_data_file, test_label_file, task ='b')

       elif task == 'c':
           if self.train:
               file = self.data_dir + '/olid-training-v1.0.tsv'
               self.label, self.tweet = ReadTrainDataset(file, task='c')       
           elif not self.train:
               test_data_file = self.data_dir + '/testset-levelc.tsv'
               test_label_file = self.data_dir + '/labels-levelc.csv'
               self.label, self.tweet = ReadTestDataset(test_data_file, test_label_file, task ='c')                  
       else:
           raise ValueError('value not support')

       
   def __len__(self):
       return len(self.label)
   def __getitem__(self, idx):

       return self.tweet[idx], self.label[idx]

##################
#done
def sampleEvalBatch():
     data = OLIDData(train = True, task = 'a')
     train_loader = DataLoader(data,\
        batch_size = 50, shuffle=True)
     ex = enumerate(train_loader)
     idx, (tweets, label) = next(ex)
     
     emoji_embedding = EmojiEmbedding(emoji_embedding_file)
     word_embedding = readEmbedding(embedding_file = embedding_file)
     
     return tweets, label 
################## 

#done
def readEmbedding(embedding_file = embedding_file):
    #read embedding from file to dictionary
    
    print("Loading Embedding Model")

    model = {}
    with open(embedding_file,encoding='UTF-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("readEmbedding Done.")
    return model
#done
def EmojiEmbedding(emoji_embedding_file):
    #must read emoji embedding using this file
    #read emoji embedding and reduce to 200-dimensional
    emoemb = readEmbedding(embedding_file = emoji_embedding_file)
    emoemb.pop('1661')    
    dfemo = pd.DataFrame.from_dict(emoemb).transpose()
    pca = PCA(n_components=200)
    emo_200 = pca.fit_transform(dfemo)
    dfemo_200 = pd.DataFrame(emo_200, index=dfemo.index)
    
    return dfemo_200.T.to_dict(('list'))

#done
def getWordVec(word, word_embedding, emoji_embedding):
    #given a word, find its GloVe vector in GloVe
    if word in word_embedding: 
        return word_embedding[word]
    elif word in emoji_embedding:
        return np.array(emoji_embedding[word])
    else:
        word_embedding[word] = np.random.rand(1, embedding_dim)       
        return word_embedding[word]
    
def class2onehot(batch_vec, num_classes = 3):
    #in: batch_size*1, out:batch_size*num_classes
    np_batch_vec = batch_vec.numpy()
    batch_size = np_batch_vec.size
    vec = np.zeros((batch_size, num_classes))
    vec[np.arange(batch_size), np_batch_vec] = 1
    return torch.from_numpy(vec)
    
    
    
def onehot2class(vec):
    #in: batch_size*num_classes, out:batch_size*1
    return vec.argmax(axis = 1)   #class start with 1

