# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:27:44 2020

@author: funrr
"""
from __future__ import print_function
import os
os.chdir(r"C:\Users\funrr\Desktop\NLP\ASSIGNMENT_2")
import Models
import util
import training
from torch.utils.data import DataLoader
import numpy as np
import torch



work_dir = os.getcwd()
batch_size = 100

def accuracy(output_1hot, label_1hot):
    #output: batch_size*num_classes, label:batch_size**num_classes (tensor or numpy)
    batch_size = label_1hot.size()[0]
    output = util.onehot2class(output_1hot)
    label = util.onehot2class(label_1hot)
    return (output == label).sum().item()/batch_size


def test(net, test_iter, device):
    #net:model, cla_label:batch_size*1, senA, senB: batch_size*sentence
    #output: accuracy in batch
    _, (cla_label,  _, senA, senB) = next(test_iter)
    net.eval()
    cla_label = util.class2onehot(cla_label).float().to(device)    
    output = net.forward(senA, senB)   #batch_size*3
    return accuracy(output, cla_label)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using CUDA.., #devices:' + str(torch.cuda.device_count()))

embedding_model = util.readEmbedding()  ##
eval_data = util.SICKData('test')
eval_loader = DataLoader(eval_data, batch_size = batch_size, shuffle=True, drop_last=True)


checkpoint = r'C:\Users\funrr\Desktop\NLP\checkpoint_199_20200303_152635.torch'
net = Models.biRNN(embedding_model, batch_size = 100, hidden_size = 50, embedding_dim = 300, dropout = 0)   
net.to(device)
if checkpoint:
    print('==> Resuming from checkpoint..')
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    net.load_state_dict(checkpoint['net_state_dicts'])
    torch_rng_state = checkpoint['torch_rng_state']
    torch.set_rng_state(torch_rng_state)
    numpy_rng_state = checkpoint['numpy_rng_state']
    np.random.set_state(numpy_rng_state)


eval_iter = enumerate(eval_loader)  
for i in range(100):
    net.eval()
    print(training.test(net, eval_iter, device))

batch_size = 100
eval_data = util.SICKData('validation')
eval_loader = DataLoader(eval_data, batch_size = batch_size, shuffle=True, drop_last=True)
eval_iter = enumerate(eval_loader)
step, (cla_label, reg_label, senA, senB) = next(eval_iter)
net.eval()
output_1hot =net.forward(senA, senB)
output = util.onehot2class(output_1hot)
(output == cla_label.to(device)).sum().item()/100