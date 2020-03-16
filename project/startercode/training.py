# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:44:51 2020

@author: funrr
"""

from __future__ import print_function
import argparse
import os
os.chdir(r"C:\Users\funrr\Desktop\NLP\ASSIGNMENT_2")
import datetime

import timeit
import Models
import matplotlib.pyplot as plt
import util
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from statistics import mean 
from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

from datetime import timedelta
work_dir = os.getcwd()

def checkpoint(epoch, net):
# Save checkpoint.
    print('Saving..')
    state = {
        'net_state_dicts': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state()
    }
    torch.save(state, work_dir + '/checkpoint_' + str(epoch) + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.torch')


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at specific epochs"""
    lr = args.lr

    if args.decay_learning_rate:
        if epoch >= 128:
            lr /= 10
        if epoch >= 180:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output_1hot, label):
    #output: batch_size*num_classes, label:batch_size*1 (tensor or numpy)
    batch_size = label.size()[0]
    output = util.onehot2class(output_1hot)

    return (output == label).sum().item()/batch_size


def test(net, test_iter, device):
    #net:model, cla_label:batch_size*1, senA, senB: batch_size*sentence
    #output: accuracy in batch
    _, (cla_label,  _, senA, senB) = next(test_iter)
    net.eval()
    cla_label = cla_label.to(device)    
    output = net.forward(senA, senB)   #batch_size*3
    return accuracy(output, cla_label)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CSI5386 training')
    parser.add_argument('--task', default="classification", type=str,
                      help='task type classification/regression (default: classification)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument("--dropout", default=0, type=float,
                        help="probability of dropout between [0, 1]")
    parser.add_argument('--batch_size', default=250, type=int, help='batch size')
    parser.add_argument('--hidden_size', default=64, type=int, help='size of lstm hidden state')
    parser.add_argument("--decay_learning_rate", help="use experimental decay learning rate", action="store_true")
    parser.add_argument('--checkpoint', type=str,                        
                        help='checkpoint from which to resume a simulation')
    parser.add_argument('--model', default="biRNN", type=str,
                      help='model type (default: biRNN)')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--epoch', default=250, type=int,
                        help='total epochs to run (including those run in previous checkpoint)')
    parser.add_argument('--weight_decay', default= 0, type=float,
                        help='weight_decay')
    args = parser.parse_args()
    
    print(datetime.datetime.now().strftime("START SIMULATION: %Y-%m-%d %H:%M"))
    sim_time_start = timeit.default_timer()


    print("ARGUMENTS:")
    for arg in vars(args):

        print(arg, getattr(args, arg))
        
        


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('Using CUDA.., #devices:' + str(torch.cuda.device_count()))
    
    train_data = util.SICKData('train')

    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, drop_last=True)
    
    test_data = util.SICKData('test')
    #test_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle=True, drop_last=True)
    if not 'embedding_model' in locals():
        embedding_model = util.readEmbedding()  ##
 

 

    # Model
    print('==> Building model..')
    
    if args.model == 'biRNN':
        if args.task == 'classification':
           net = Models.biRNN(embedding_model, batch_size = args.batch_size, hidden_size = args.hidden_size, embedding_dim = 300, dropout = args.dropout)   
        elif args.task == 'regression':
           net = Models.biRNN(embedding_model, batch_size = args.batch_size, hidden_size = args.hidden_size, embedding_dim = 300, dropout = args.dropout, regression=True)    
    if args.model =='CNN':
        if args.task == 'classification':
           net = Models.CNN(embedding_model, dropout = args.dropout)   
        elif args.task == 'regression':
           net = Models.CNN(embedding_model, dropout = args.dropout, regression = True)    
        
    #TODO: add regression CNN 
    net.to(device)
 
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.task == 'classification':
        loss_fun = nn.CrossEntropyLoss()
    elif args.task == 'regression':
        loss_fun = nn.MSELoss()

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    # Load checkpoint.
    if args.checkpoint:
        print('==> Resuming from checkpoint..')
        print(args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net_state_dicts'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        torch_rng_state = checkpoint['torch_rng_state']
        torch.set_rng_state(torch_rng_state)
        numpy_rng_state = checkpoint['numpy_rng_state']
        np.random.set_state(numpy_rng_state)


    #train_iter = enumerate(train_loader)  
    #step, (cla_label, reg_label, senA, senB) = next(train_iter)
    



    train_acc= []
    test_acc = []
    if args.task == 'classification':
        result = {'Accuracy':[],\
                  'Precision0':[],'Precision1':[],'Precision2':[],\
                  'Recall0':[],'Recall1':[],'Recall2':[],\
                  'F0':[],'F1':[],'F2':[]}
    elif args.task == 'regression':
        result = {'pearson':[], 'MSE':[],'Spearman':[]}

    
    for epoch in range(start_epoch, args.epoch):      
        test_iter = enumerate(test_loader)   
        for step, (cla_label, reg_label, senA, senB) in enumerate(train_loader):  
            net.train()
            #cla_label = util.class2onehot(cla_label).float().to(device)  
            if args.task == 'classification':
                label = cla_label.long().to(device)
            elif args.task == 'regression':
                label = reg_label.float().to(device)
            optimizer.zero_grad()       
            output = net.forward(senA, senB)   #batch_size*3
            loss = loss_fun(output, label)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            
            #print progress every 20 steps
            if step%20 == 0:
                if args.task == 'classification':
                    if len(train_acc) > 0 and len(test_acc) > 0:
                        print("epoch : % 3d,  iter: % 5d,  loss:% .4f,  train acc:% .4f,  test acc:% .4f" %(epoch+1, step+1, loss.item(), train_acc[len(train_acc)-1], test_acc[len(test_acc)-1]))
                else:
                    print("epoch : % 3d,  iter: % 5d,  loss:% .4f" %(epoch+1, step+1, loss.item()))
            if args.task == 'classification':
            #plot training curve every every 100 steps
                if (step)%100 == 0:    
                    train_acc.append(accuracy(output, label))
                    with torch.no_grad():
        
                        test_acc.append(test(net, test_iter, device))
                        optimizer.zero_grad() 
        
        
                    fig1 = plt.figure()
                    ax1 = fig1.add_subplot(111)
                    ax1.plot(train_acc, label='training acc')
                    ax1.plot(test_acc, label='test acc')
                    ax1.legend(prop={'size': 9})
                    ##############################################################
                    title = "CNN training curve"
                    ax1.set_title(title)
                    ax1.set_xlabel("train steps")
                    ax1.set_ylabel("accuracy")
                    plt.pause(0.05)
                    fig1   
                    
            if epoch >= (args.epoch-5):
                #last 5 epochs, calculate evaluation result in each step
                test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle=False, drop_last=True)
                net.eval()
                for step, (cla_label, reg_label, senA, senB) in enumerate(test_loader):        
                    with torch.no_grad():
                        output = net.forward(senA, senB)
                    if args.task == 'classification':
                        output = output.argmax(dim = 1).detach().cpu().numpy()
                        cla_label = cla_label.detach().cpu().numpy()
                        acc = accuracy_score(cla_label, output)
                        prf = precision_recall_fscore_support(cla_label, output)
                        result['Accuracy'].append(acc)
                        result['Precision0'].append(prf[0][0])
                        result['Precision1'].append(prf[0][1])
                        result['Precision2'].append(prf[0][2])
                        result['Recall0'].append(prf[1][0])
                        result['Recall1'].append(prf[1][1])
                        result['Recall2'].append(prf[1][2])
                        result['F0'].append(prf[2][0])
                        result['F1'].append(prf[2][1])
                        result['F2'].append(prf[2][2])
                        

                    elif args.task == 'regression':
                        output = output.detach().cpu().numpy().flatten()
                        reg_label = reg_label.detach().cpu().numpy()
                        mse = mean_squared_error(reg_label, output)

                        result['MSE'].append(mean_squared_error(reg_label, output))
                        result['pearson'].append(pearsonr(reg_label,output)[0])
                        result['Spearman'].append(spearmanr(reg_label,output)[0])

                        
            
            


                

             
            
        adjust_learning_rate(optimizer, epoch)


        #save model every 10 epoches, and the last 3 epoch
        if (epoch-start_epoch)%50 == 0 or epoch >= (args.epoch-3):
            checkpoint_time_start = timeit.default_timer()
            checkpoint(epoch, net)
            checkpoint_time_end = timeit.default_timer()
            elapsed_seconds = round(checkpoint_time_end - checkpoint_time_start)
            print('Checkpoint Saving, Duration (Hours:Minutes:Seconds): ' + str(timedelta(seconds=elapsed_seconds)))

    # Print elapsed time and current time
    elapsed_seconds = round(timeit.default_timer() - sim_time_start)
    print('Simulation Duration (Hours:Minutes:Seconds): ' + str(timedelta(seconds=elapsed_seconds)))
    print(datetime.datetime.now().strftime("END SIMULATION: %Y-%m-%d %H:%M"))













