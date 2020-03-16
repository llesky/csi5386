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

ME_DIR = os.path.dirname(os.path.realpath(__file__))

work_dir = ME_DIR
embedding_file = work_dir + '/word_embedding/glove.6B.300d.txt'
num_classes = 3
embedding_dim = 300



    

def sentenceTransform(sentenceList):
    #gives a list of sentences, return a cleaned, tokenized sentenceList, and calculate the longest sentence length
    token_sentence_list = []

    for sentence in sentenceList:
        sentence = sentence.lower()      #Tolowercase    
        sentence = tokenize.wordpunct_tokenize(sentence)   #tokenize 
        token_sentence_list.append(sentence)

        

    return token_sentence_list

def readDataset(file):
    data = pd.read_csv(file, sep='\t')
    data.replace(['ENTAILMENT','CONTRADICTION', 'NEUTRAL'], [0, 1, 2], inplace=True)
    cla_label = data.iloc[:,4].to_numpy()
    reg_label = data.iloc[:,3].to_numpy()
    #senA = sentenceTransform(data.iloc[:,1].tolist())
    #senB = sentenceTransform(data.iloc[:,2].tolist())
    senA = data.iloc[:,1].tolist()
    senB = data.iloc[:,2].tolist()
    
    return cla_label, reg_label, senA, senB
    
class SICKData(Dataset):
   #SICK data class; used by imdbDataLoader
   #data: either 'train', 'validation' or 'test'
   
   def __init__(self, data='train'):
       self.newdata_dir = work_dir+'\\dataset'
       self.data = data   
       if self.data == 'train':
           file = work_dir + '/ASSIGNMENT_2/SICK_train.txt'
       elif self.data == 'validation':
           file = work_dir + '/ASSIGNMENT_2/SICK_trial.txt'
       elif self.data == 'test':
           file = work_dir + '/ASSIGNMENT_2/SICK_test_annotated.txt'
       else:
           raise ValueError('value not support')

       self.cla_label, self.reg_label, self.senA, self.senB = readDataset(file)       
   def __len__(self):
       return len(self.cla_label)
   def __getitem__(self, idx):
       cla_label = self.cla_label[idx]
       reg_label = self.reg_label[idx]
       senA = self.senA[idx]
       senB = self.senB[idx]

       return cla_label, reg_label, senA, senB
##################
def sampleEvalBatch():
     dat = SICKData('train')
     train_loader = DataLoader(dat,\
        batch_size = 50, shuffle=True)
     ex = enumerate(train_loader)
     b = next(ex)
     return b 
################## 
def readEmbedding(embedding_file = embedding_file):
    print("Loading Glove Model")

    model = {}
    with open(embedding_file,encoding='UTF-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("readEmbedding Done.")
    return model

def getWordVec(word, embedding):
    #given a word, find its GloVe vector in GloVe
    if word in embedding:
        return embedding[word]
    else:
        embedding[word] = np.random.rand(1, embedding_dim)
       
        return embedding[word]
    
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
    
