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

output_size = 3


#############
def testscript():
    dat = util.SICKData('train')
    batch = util.sampleEvalBatch()
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

 
    def EmbeddingPadding(self, senA, senB, max_length = 0, fit_batch = True):
        #in: senA, senB, tuples of tokenized sentences of batch_size      
        #out:embedded and padded, shape[batch_size, max_len_of_word, embedding_dim]          
        #if  fit_batch, then max_length equals the longest sentence in the batch
        
        senA_list = []      #vectoriezed senA
        senB_list = []      #vectoriezed senB
        batch_size = len(senA)
        senA = util.sentenceTransform(senA)
        senB = util.sentenceTransform(senB)
        for sentence in senA:
            sentence_len = len(sentence)
            sentence_vec = np.zeros([sentence_len, self.embedding_dim])
            for idx, word in enumerate(sentence):
                sentence_vec[idx,:] = util.getWordVec(word, self.embedding_model).reshape(1,-1)
            senA_list.append(torch.from_numpy(sentence_vec))
        if not fit_batch:
            senA_list.append(torch.zeros(max_length, self.embedding_dim))                        
        senA_list = torch.nn.utils.rnn.pad_sequence(senA_list, batch_first=True)
        senA_list = senA_list[:batch_size, :,:]
        
        for sentence in senB:
            sentence_len = len(sentence)
            sentence_vec = np.zeros([sentence_len, self.embedding_dim])
            for idx, word in enumerate(sentence):
                sentence_vec[idx,:] = util.getWordVec(word, self.embedding_model).reshape(1,-1)
            senB_list.append(torch.from_numpy(sentence_vec))
        if not fit_batch:
            senB_list.append(torch.zeros(max_length, self.embedding_dim))                        
        senB_list = torch.nn.utils.rnn.pad_sequence(senB_list, batch_first=True)
        senB_list = senB_list[:batch_size, :,:]

        

        return senA_list.float().to(self.device), senB_list.float().to(self.device)
        

class biRNN(model):              
        
    def __init__(self, embedding_model, batch_size = 50, hidden_size = 20, embedding_dim = 300, dropout = 0, regression = False):  
        super().__init__(embedding_model, embedding_dim = embedding_dim)
        #not tuneable
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.num_layer = 3
        #tuneable
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.regression = regression
        self.mlp_structure = [50, 20]

        
        self.LSTM_A = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = 3, \
                              batch_first = True, dropout = self.dropout, bidirectional  = True)
        self.LSTM_B = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_size, num_layers = 3, \
                         batch_first = True, dropout = self.dropout, bidirectional  = True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2*4, self.mlp_structure[0], bias= True),     
            nn.ReLU(),
            nn.Linear(self.mlp_structure[0], self.mlp_structure[1]),
            nn.ReLU(),
            nn.Linear(self.mlp_structure[1], 3)
           # nn.ReLU(),
            #nn.Linear(self.mlp_structure[2], 3),
            )
        
        if self.regression == True:
            self.to_scalar = nn.Linear(3, 1)
        else:    
            self.softmax = nn.Softmax()
       
    def forward(self, senA, senB):
        #senA, senB: [batch_size, max_len_of_word, embedding_dim]     
        h0_A = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        c0_A = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        h0_B = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        c0_B = torch.randn(self.num_layer*2, self.batch_size, self.hidden_size).to(self.device)
        senA, senB = self.EmbeddingPadding(senA, senB, fit_batch = True) 
        
        output_A, _ = self.LSTM_A(senA, (h0_A, c0_A))
        output_B, _ = self.LSTM_B(senB, (h0_B, c0_B))
        

        
        encoding = torch.cat((output_A.mean(dim = 1), (output_A.mean(dim = 1)-output_B.mean(dim = 1)).abs(), output_A.mean(dim = 1)*output_B.mean(dim = 1), output_B.mean(dim = 1)), dim=1)  #concatenate mean pool of two lstm, then pass to a MLP

        output = self.fc(encoding)
        if self.regression is True:
            output = self.to_scalar(output)
        
        else:
            output = self.softmax(output)
        
        return output


     
   
class CNN(model):
    def __init__(self, embedding_model,  num_channels = 100, kernel_size = [2,3,4], max_sen_len = 30, batch_size = 100 , embedding_dim = 300, dropout = 0.5, regression = False):  
        super().__init__(embedding_model, embedding_dim = embedding_dim)
        #not tuneable
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        #tuneable
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len

        self.dropout_rate = dropout
        self.regression = regression

      
        self.convA1 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_channels, kernel_size=self.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size[0]+1)
        )
        self.convA2 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_channels, kernel_size=self.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size[1]+1)
        )
        self.convA3 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_channels, kernel_size=self.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size[2]+1)
        )
        
        self.convB1 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_channels, kernel_size=self.kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size[0]+1)
        )
        self.convB2 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_channels, kernel_size=self.kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size[1]+1)
        )
        self.convB3 = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_channels, kernel_size=self.kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - self.kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        if not self.regression:
        # Fully-Connected Layer
            self.fc = nn.Linear(self.num_channels*len(self.kernel_size)*2, 3)
        else:
            self.fc = nn.Linear(self.num_channels*len(self.kernel_size)*2, 1)
        # Softmax non-linearity
        self.softmax = nn.Softmax()
        
    def forward(self, senA, senB):
        # x.shape = (max_sen_len, batch_size)
        senA, senB = self.EmbeddingPadding(senA, senB, max_length = self.max_sen_len, fit_batch = False) 
                #shape[batch_size, max_len_of_word, embedding_dim]     
        senA = senA.transpose(1, 2)
        senB = senB.transpose(1, 2)
                # embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)
     

        conv_outA1 = self.convA1(senA).squeeze(2) #shape=(64, num_channels, 1) (squeeze 1)
        conv_outA2 = self.convA2(senA).squeeze(2)
        conv_outA3 = self.convA3(senA).squeeze(2)
        conv_outB1 = self.convB1(senB).squeeze(2) #shape=(64, num_channels, 1) (squeeze 1)
        conv_outB2 = self.convB2(senB).squeeze(2)
        conv_outB3 = self.convB3(senB).squeeze(2)
        
        all_outA = torch.cat((conv_outA1, conv_outA2, conv_outA3), 1)
        all_outB = torch.cat((conv_outB1, conv_outB2, conv_outB3), 1)
        all_out = torch.cat((all_outA, all_outB), 1)
        final_feature_map = self.dropout(all_out)     
        final_out = self.fc(final_feature_map)
        if not self.regression:
            final_out = self.softmax(final_out)
        
        return final_out