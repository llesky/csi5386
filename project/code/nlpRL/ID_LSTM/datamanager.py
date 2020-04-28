import os
from nltk import tokenize
import numpy as np
import tensorflow as tf
import json, random
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA

ME_DIR = os.path.dirname(os.path.realpath(__file__))

work_dir = ME_DIR
embedding_file = work_dir + '/embedding/glove.twitter.27B.200d.txt'
emoji_embedding_file = work_dir + '/embedding/emoji2vec.txt'

train_file = work_dir + '/OLIDv1.0/olid-training-v1.0.tsv'
test_label_file = work_dir + '/OLIDv1.0/labels-levela.csv'
test_data_file = work_dir + '/OLIDv1.0/testset-levela.tsv'

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


class DataManager(object):
    def __init__(self, task = 'a'):
        '''
        Read the data from dir "dataset"
        '''
        self.data_dir = work_dir+'/OLIDv1.0'    
        self.file = ""
        self.test_data_file = ""
        if task == 'a':
            self.file = self.data_dir + '/olid-training-v1.0.tsv'
            self.label_train, self.tweet_train = ReadTrainDataset(self.file, task='a')       
            self.test_data_file = self.data_dir + '/testset-levela.tsv'
            test_label_file = self.data_dir + '/labels-levela.csv'
            self.label_test, self.tweet_test = ReadTestDataset(test_data_file, test_label_file, task ='a')
    
        elif task == 'b':
            self.file = self.data_dir + '/olid-training-v1.0.tsv'
            self.label_train, self.tweet_train = ReadTrainDataset(self.file, task='b')       
            self.test_data_file = self.data_dir + '/testset-levelb.tsv'
            test_label_file = self.data_dir + '/labels-levelb.csv'
            self.label_test, self.tweet_test = ReadTestDataset(test_data_file, test_label_file, task ='b')

        elif task == 'c':
            self.file = self.data_dir + '/olid-training-v1.0.tsv'
            self.label_train, self.tweet_train = ReadTrainDataset(self.file, task='c')       
            self.test_data_file = self.data_dir + '/testset-levelc.tsv'
            test_label_file = self.data_dir + '/labels-levelc.csv'
            self.label_test, self.tweet_test = ReadTestDataset(test_data_file, test_label_file, task ='c')                  
        else:
            raise ValueError('value not support')
        
        self.tweet_test = sentenceTransform(self.tweet_test)
        self.tweet_train = sentenceTransform(self.tweet_train)
    def getword(self):
        '''
        Get the words that appear in the data.
        Sorted by the times it appears.
        {'ok': 1, 'how': 2, ...}
        Never run this function twice.
        '''
        wordcount = {}
        #allwords = np.append(self.tweet_train,self.tweet_test)
        self.alllabel = np.append(self.label_train,self.label_test)
        words = []
        #print(allwords)
        for i in self.tweet_train:
            for x in i:
                words.append(x)
        for i in self.tweet_test:
            for x in i:
                words.append(x)
        words = Counter(words).most_common()
        # sorted(words,key = lambda x : x[1], reverse = True)
        #self.wordlist = words
        self.wordlist = {item[0]: index+1 for index, item in enumerate(words)}
        #print(self.wordlist)
        return self.wordlist
    
    def getdata(self, grained, maxlenth):
        '''
        Get all the data, divided into (train,dev,test).
        For every sentence, {'words':[1,3,5,...], 'solution': [0,1,0,0,0]}
        For each data, [sentence1, sentence2, ...]
        Never run this function twice.
        '''
        def one_hot_vector(r):
            s = np.zeros(grained, dtype=np.float32)
            s[r] += 1.0
            return s
        self.getword()
        self.data = {}
        for fname in ['train', 'dev', 'test']:
            self.data[fname] = []
            if fname=='train':
                allwords = self.tweet_train
                alllabel = self.label_train
            else:
                allwords = self.tweet_test
                alllabel = self.label_test
            for index,line in enumerate(allwords):
                words = []
                for i in line:
                    word = self.wordlist[i.lower()]
                    words.append(word)
                lens = len(words)
                if maxlenth < lens:
                    print(lens)
                words += [0] * (maxlenth - lens)
                solution = one_hot_vector(int(alllabel[index]))
                now = {'words': np.array(words), \
                        'solution': solution,\
                        'lenth': lens}
                self.data[fname].append(now)
        return self.data['train'], self.data['dev'], self.data['test']
    
    def get_wordvector(self, name,emoji_name=None):
        self.wv = {}
        if emoji_name != None:
            fr = open(name)

            emoemb = readEmbedding(embedding_file = emoji_name)
            #emoemb.pop('1661')
            dfemo = pd.DataFrame.from_dict(emoemb).transpose()
            pca = PCA(n_components=200)
            emo_200 = pca.fit_transform(dfemo)
            dfemo_200 = pd.DataFrame(emo_200, index=dfemo.index)
            er = dfemo_200.T.to_dict(('list'))

            while fr.readline():
                vec = fr.readline().split()
                word = vec[0].lower()
                vec = np.array(vec[1:],dtype=np.float32)
                if word in self.wordlist:
                    self.wv[self.wordlist[word]] = vec

            for key, value in er.items(): 
                vec = np.array(value,dtype=np.float32)
                if key in self.wordlist:
                    self.wv[self.wordlist[key]] = vec
        else:
            with open(name, 'r') as f: # Again temporary file for reading
                d = {}
                l = f.read().split('\n')      # Split using commas
                for i in l:
                    values = i.split(',')   # Split using ': '
                    if len(values[1:]) > 1024:
                        #print(len(values[1:]))
                        d[(values[0]+values[1])] = values[2:] # Any type conversion will need to happen here
                    else:
                        d[values[0]] = values[1:]
                    #print(values[0])

            # dfemo = pd.DataFrame.from_dict(d).transpose()
            # pca = PCA(n_components=200)
            # emo_200 = pca.fit_transform(dfemo)
            # dfemo_200 = pd.DataFrame(emo_200, index=dfemo.index)
            # er = dfemo_200.T.to_dict(('list'))

            for key, value in d.items(): 
                vec = np.array(value,dtype=np.float32)
                if key in self.wordlist:
                    self.wv[self.wordlist[key]] = vec
        
        self.wordvector = []
        losscnt = 0
        for i in range(len(self.wordlist) + 1):
            if i in self.wv:
                self.wordvector.append(self.wv[i])
            else:
                losscnt += 1
                self.wordvector.append(np.random.uniform(-0.1,0.1,[200]))
        self.wordvector = np.array(self.wordvector, dtype=np.float32)
        print (losscnt, "words not find in wordvector")
        print (len(self.wordvector), "words in total")
        return self.wordvector


# print(train_data[0])
# print(dev_data[0])
# print(test_data[0])



# wv = datamanager.get_wordvector("../WordVector/vector.25dim")
# mxlen = 0
# for item in train_data:
#    print(item['lenth'])
#    if item['lenth'] > mxlen:
#        mxlen =item['lenth']
# print(mxlen)
