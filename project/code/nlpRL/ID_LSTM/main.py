import numpy as np
import tensorflow as tf
import random
import sys, os
import json
import argparse
import torch

from parser import Parser
from datamanager import DataManager
from actor import ActorNetwork
from LSTM_critic import LSTM_CriticNetwork
from torch.utils.data import DataLoader

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#get parse
argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
random.seed(args.seed)

#get data

ME_DIR = os.path.dirname(os.path.realpath(__file__))
work_dir = ME_DIR
embedding_file = work_dir + '/embedding/glove.twitter.27B.200d.txt'
emoji_embedding_file = work_dir + '/embedding/emoji2vec.txt'
embedding_file_ = work_dir+'/embedding/dict_file.csv'
embedding_dim = 200

datamanager = DataManager('a')
train_data, test_data, dev_data = datamanager.getdata(2, args.maxlenth)
word_vector = datamanager.get_wordvector(embedding_file,emoji_embedding_file)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_data = util.OLIDData(train=True, task = 'a')
# test_data = util.OLIDData(train=False, task = 'a')
# train_loader = DataLoader(train_data, batch_size = args.batchsize, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_data, batch_size = args.batchsize, shuffle=True, drop_last=True)

'''
if not 'word_embedding' in locals():
    word_embedding = util.readEmbedding(embedding_file)  #read embedding models
if not 'emoji_embedding' in locals():
    emoji_embedding = util.EmojiEmbedding(emoji_embedding_file)  ##'''

def structure_data(data):
    d = []
    for i in data:
        data_ = {}
        #data_["words"] = EmbeddingPadding(i[0])
        data_["words"] = i[0]
        data_["solution"] = i[1]
        data_["lenth"] = len(i[0])
        d.append(data_)
    return d

# train_data = structure_data(train_data)
# test_data = structure_data(test_data)
# dev_data = test_data



def sampling_RL(sess, actor, inputs, vec, lenth, epsilon=0., Random=True):
    #print epsilon
    current_lower_state = np.zeros((1, 2*args.dim), dtype=np.float32)
    actions = []
    states = []
    #sampling actions

    for pos in range(lenth):
        predicted = actor.predict_target(current_lower_state, [vec[0][pos]])
        
        states.append([current_lower_state, [vec[0][pos]]])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < predicted[0] else 1)
            else:
                action = (1 if random.random() < predicted[0] else 0)
        else:
            action = np.argmax(predicted)
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = critic.lower_LSTM_target(current_lower_state, [[inputs[pos]]])
    
    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            Rinput.append(inputs[i])
    Rlenth = len(Rinput)
    if Rlenth == 0:
        actions[lenth-2] = 1
        Rinput.append(inputs[lenth-2])
        Rlenth = 1
    Rinput += [0] * (args.maxlenth - Rlenth)
    return actions, states, Rinput, Rlenth

def train(sess, actor, critic, train_data, batchsize, samplecnt=5, LSTM_trainable=True, RL_trainable=True):
    print ("training : total ", len(train_data), "nodes.")
    random.shuffle(train_data)
    for b in range(int(len(train_data) / batchsize)):
        datas = train_data[b * batchsize: (b+1) * batchsize]
        totloss = 0.
        critic.assign_active_network()
        actor.assign_active_network()
        for j in range(batchsize):
            #prepare
            data = datas[j]
            inputs, solution, lenth = data['words'], data['solution'], data['lenth']
            #train the predict network
            if RL_trainable:
                actionlist, statelist, losslist = [], [], []
                aveloss = 0.
                for i in range(samplecnt):
                    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, args.epsilon, Random=True)
                    actionlist.append(actions)
                    statelist.append(states)
                    out, loss = critic.getloss([Rinput], [Rlenth], [solution])
                    loss += (float(Rlenth) / lenth) **2 *0.15
                    aveloss += loss
                    losslist.append(loss)
                
                aveloss /= samplecnt
                totloss += aveloss
                grad = None
                if LSTM_trainable:
                    out, loss, _ = critic.train([Rinput], [Rlenth], [solution])
                for i in range(samplecnt):
                    for pos in range(len(actionlist[i])):
                        rr = [0., 0.]
                        rr[actionlist[i][pos]] = (losslist[i] - aveloss) * args.alpha
                        g = actor.get_gradient(statelist[i][pos][0], statelist[i][pos][1], rr)
                        if grad == None:
                            grad = g
                        else:
                            grad[0] += g[0]
                            grad[1] += g[1]
                            grad[2] += g[2]
                actor.train(grad)
            else:
                out, loss, _ = critic.train([inputs], [lenth], [solution])
                totloss += loss
        if RL_trainable:
            actor.update_target_network()
            if LSTM_trainable:
                critic.update_target_network()
        else:
            critic.assign_target_network()
        if (b + 1) % 500 == 0:
            acc_test = test(sess, actor, critic, test_data, noRL= not RL_trainable)
            acc_dev = test(sess, actor, critic, dev_data, noRL= not RL_trainable)
            print ("batch ",b , "total loss ", totloss, "----test: ", acc_test, "| dev: ", acc_dev)


def test(sess, actor, critic, test_data, noRL=False):
    acc = 0
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth = data['words'], data['solution'], data['lenth']
        #predict
        if noRL:
            out = critic.predict_target([inputs], [lenth])
        else:
            actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, Random=False)
            out = critic.predict_target([Rinput], [Rlenth])
        if np.argmax(out) == np.argmax(solution):
            acc += 1
    return float(acc) / len(test_data)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    #model
    critic = LSTM_CriticNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.maxlenth, args.dropout, word_vector) 
    actor = ActorNetwork(sess, args.dim, args.optimizer, args.lr, args.tau)
    #print variables
    for item in tf.trainable_variables():
        print (item.name, item.get_shape())
    
    saver = tf.train.Saver()
    
    #LSTM pretrain
    if args.RLpretrain != '':
        pass
    elif args.LSTMpretrain == '':
        sess.run(tf.global_variables_initializer())
        for i in range(0, 2):
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, RL_trainable=False)
            critic.assign_target_network()
            acc_test = test(sess, actor, critic, test_data, True)
            acc_dev = test(sess, actor, critic, dev_data, True)
            print("LSTM_only ",i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_base", global_step=i)
        print("LSTM pretrain OK")
    else:
        print("Load LSTM from ", args.LSTMpretrain)
        saver.restore(sess, args.LSTMpretrain)
    
    print("epsilon", args.epsilon)

    if args.RLpretrain == '':
        for i in range(0, 5):
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, LSTM_trainable=False)
            acc_test = test(sess, actor, critic, test_data)
            acc_dev = test(sess, actor, critic, dev_data)
            print("RL pretrain ", i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_RLpre", global_step=i)
        print("RL pretrain OK")
    else:
        print("Load RL from", args.RLpretrain)
        saver.restore(sess, args.RLpretrain)

    for e in range(args.epoch):
        train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt)
        acc_test = test(sess, actor, critic, test_data)
        acc_dev = test(sess, actor, critic, dev_data)
        print("epoch ", e, "----test: ", acc_test, "| dev: ", acc_dev)
        saver.save(sess, "checkpoints/"+args.name, global_step=e)

