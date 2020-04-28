import numpy as np
import tensorflow as tf
import json, random

class DataManager(object):
    def __init__(self, dataset):
        '''
        Read the data from dir "dataset"
        '''
        self.origin = {}
        for fname in ['train', 'dev', 'test']:
            data = []
            for line in open('%s/%s.res' % (dataset, fname)):
                s = json.loads(line.strip())
                if len(s) > 0:
                    data.append(s)
            self.origin[fname] = data
            
    def getword(self):
        '''
        Get the words that appear in the data.
        Sorted by the times it appears.
        {'ok': 1, 'how': 2, ...}
        Never run this function twice.
        '''
        wordcount = {}
        def dfs(node):
            if 'children' in node:
                dfs(node['children'][0])
                dfs(node['children'][1])
            else:
                word = node['word'].lower()
                wordcount[word] = wordcount.get(word, 0) + 1
        for fname in ['train', 'dev', 'test']:
            for sent in self.origin[fname]:
                dfs(sent)
        words = wordcount.items()
        sorted(words,key = lambda x : x[1], reverse = True)
        self.words = words
        self.wordlist = {item[0]: index+1 for index, item in enumerate(words)}
        print(self.wordlist,"aaa")
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
        def dfs(node, words):
            if 'children' in node:
                dfs(node['children'][0], words)
                dfs(node['children'][1], words)
            else:
                word = self.wordlist[node['word'].lower()]
                words.append(word)
        self.getword()
        self.data = {}
        for fname in ['train', 'dev', 'test']:
            self.data[fname] = []
            for sent in self.origin[fname]:
                words = []
                dfs(sent, words)
                lens = len(words)
                if maxlenth < lens:
                    print(lens)
                words += [0] * (maxlenth - lens)
                solution = one_hot_vector(int(sent['rating']))
                now = {'words': np.array(words), \
                        'solution': solution,\
                        'lenth': lens}
                self.data[fname].append(now)
        #print(self.data['train'])
        return self.data['train'], self.data['dev'], self.data['test']
    
    def get_wordvector(self, name,emoji_name):
        fr = open(name)
        er = open(emoji_name)
        self.wv = {}
        while fr.readline():
            vec = fr.readline().split()
            word = vec[0].lower()
            vec = np.array(vec[1:],dtype=np.float32)
            if word in self.wordlist:
                self.wv[self.wordlist[word]] = vec
        while er.readline():
            vec = er.readline().split()
            if len(vec)<1:
                break
            word = vec[0]
            vec = np.array(vec[1:],dtype=np.float32)
            if word in self.wordlist:
                self.wv[self.wordlist[word]] = vec
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
