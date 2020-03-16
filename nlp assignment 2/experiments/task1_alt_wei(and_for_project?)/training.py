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


import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

from datetime import timedelta
work_dir = os.getcwd()

def checkpoint(epoch):
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
        if epoch >= 50:
            lr /= 10
        if epoch >= 80:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CSI5386 training')
    parser.add_argument('--task', default="classfication", type=str,
                      help='task type (default: classfication)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument("--dropout_lstm", default=0, type=float,
                        help="probability of dropout between [0, 1]")
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--hidden_size', default=50, type=int, help='size of lstm hidden state')
    parser.add_argument("--decay_learning_rate", help="use experimental decay learning rate", action="store_true")
    parser.add_argument('--checkpoint', type=str,                        
                        help='checkpoint from which to resume a simulation')
    parser.add_argument('--model', default="biRNN", type=str,
                      help='model type (default: biRNN)')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--epoch', default=100, type=int,
                        help='total epochs to run (including those run in previous checkpoint)')
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
    #
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, drop_last=True)
    
    test_data = util.SICKData('test')
    #test_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle=True, drop_last=True)
    embedding_model = util.readEmbedding()  ##
 

 

    # Model
    print('==> Building model..')
    
    if args.model == 'biRNN':
         net = Models.biRNN(embedding_model, batch_size = args.batch_size, hidden_size = args.hidden_size, embedding_dim = 300, dropout = args.dropout_lstm)   
    #TODO: add more models
    net.to(device)
 
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    CEloss = nn.BCELoss()

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
    

    testloss_tick = 12

    train_acc= []
    test_acc = []

    
    for epoch in range(start_epoch, args.epoch):      
        test_iter = enumerate(test_loader)   
        for step, (cla_label, reg_label, senA, senB) in enumerate(train_loader):  
            net.train()
            cla_label = util.class2onehot(cla_label).float().to(device)            
            optimizer.zero_grad()       
            output = net.forward(senA, senB)   #batch_size*3
            loss = CEloss(output, cla_label)
            #TODO add regression option
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
            
            #print progress every 20 steps
            if step%20 == 0:
                if len(train_acc) > 0 and len(test_acc) > 0:
                    print("epoch : % 3d,  iter: % 5d,  loss:% .4f,  train acc:% .4f,  test acc:% .4f" %(epoch+1, step+1, loss.item(), train_acc[len(train_acc)-1], test_acc[len(test_acc)-1]))
           
            #plot training curve every every 100 steps
            if (step)%100 == 0:    
                train_acc.append(accuracy(output, cla_label))
                with torch.no_grad():

                    test_acc.append(test(net, test_iter, device))
                    optimizer.zero_grad() 


                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.plot(train_acc, label='training acc')
                ax1.plot(test_acc, label='test acc')
                ax1.legend(prop={'size': 9})
                ##############################################################
                title = "LSTM training curve"
                ax1.set_title(title)
                ax1.set_xlabel("train steps")
                ax1.set_ylabel("accuracy")
                plt.pause(0.05)
                fig1   

                

                

             
            
        adjust_learning_rate(optimizer, epoch)


        #save model every 5 epoches, and the last 3 epoch
        if (epoch-start_epoch)%5 == 0 or epoch >= (args.epoch-3):
            checkpoint_time_start = timeit.default_timer()
            checkpoint(epoch)
            checkpoint_time_end = timeit.default_timer()
            elapsed_seconds = round(checkpoint_time_end - checkpoint_time_start)
            print('Checkpoint Saving, Duration (Hours:Minutes:Seconds): ' + str(timedelta(seconds=elapsed_seconds)))

    # Print elapsed time and current time
    elapsed_seconds = round(timeit.default_timer() - sim_time_start)
    print('Simulation Duration (Hours:Minutes:Seconds): ' + str(timedelta(seconds=elapsed_seconds)))
    print(datetime.datetime.now().strftime("END SIMULATION: %Y-%m-%d %H:%M"))





                
 

    
 










