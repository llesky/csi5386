# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:55:43 2020

@author: funrr
"""

import os
import util
import numpy as np
import torch 
import torch.nn as nn




#############
def testscript():
    dat = util.SICKData('train')
    batch = util.sampleEvalBatch(dat)
    senA = batch[1][2] 
    senB = batch[1][3]
    embedding_model = util.readEmbedding()
    test_par_model = model(embedding_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_child_model = biRNN(embedding_model).to(device)
    test_child_model.forward(senA,senB)
#############
class model(nn.Module):              
        
    def __init__(self, embedding_model, embedding_dim = 300):  
        super().__init__()  
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
    def EmbeddingPadding(self, senA, senB):
        #in: senA, senB, tuples of tokenized sentences of batch_size      
        #out:embedded and padded, shape[batch_size, max_len_of_word, embedding_dim]          
        
    
        senA_list = []      #vectoriezed senA
        senB_list = []      #vectoriezed senB
        senA = util.sentenceTransform(senA)
        senB = util.sentenceTransform(senB)
        for sentence in senA:
            sentence_vec = np.zeros([len(sentence), self.embedding_dim])
            for idx, word in enumerate(sentence):
                sentence_vec[idx,:] = util.getWordVec(word, self.embedding_model).reshape(1,-1)
            senA_list.append(torch.from_numpy(sentence_vec))
        senA_list = torch.nn.utils.rnn.pad_sequence(senA_list, batch_first=True)
        
        for sentence in senB:
            sentence_vec = np.zeros([len(sentence), self.embedding_dim])
            for idx, word in enumerate(sentence):
                sentence_vec[idx,:] = util.getWordVec(word, self.embedding_model).reshape(1,-1)
            senB_list.append(torch.from_numpy(sentence_vec))
        senB_list = torch.nn.utils.rnn.pad_sequence(senB_list, batch_first=True)
        

        return senA_list.float().to(self.device), senB_list.float().to(self.device)
        

class biRNN(model):              
        
    def __init__(self, embedding_model, batch_size = 50, hidden_size = 20, embedding_dim = 300, dropout = 0):  
        super().__init__(embedding_model, embedding_dim = embedding_dim)
        #not tuneable
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.num_layer = 3
        #tuneable
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mlp_structure = [50, 20]

        
        self.LSTM_A = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = 3, \
                              batch_first = True, dropout = self.dropout, bidirectional  = True)
        self.LSTM_B = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = 3, \
                         batch_first = True, dropout = self.dropout, bidirectional  = True)
        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size * 2, self.mlp_structure[0], bias= True),     
            nn.ReLU(),
            nn.Linear(self.mlp_structure[0], self.mlp_structure[1]),
            nn.ReLU(),
            #nn.Linear(self.mlp_structure[1], self.mlp_structure[2]),
            #nn.ReLU(),
            nn.Linear(self.mlp_structure[1], 3),
            nn.Sigmoid()
            )
       
    def forward(self, senA, senB):
        #senA, senB: [batch_size, max_len_of_word, embedding_dim]     
        h0_A = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        c0_A = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        h0_B = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        c0_B = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        senA, senB = self.EmbeddingPadding(senA, senB) 
        
        output_A, _ = self.LSTM_A(senA, (h0_A, c0_A))
        output_B, _ = self.LSTM_B(senB, (h0_B, c0_B))
        
        encoding = torch.cat((output_A.mean(dim = 1), output_B.mean(dim = 1)), dim=1)  #concatenate mean pool of two lstm, then pass to a MLP

        output = self.fc(encoding)
        
        return output

        
