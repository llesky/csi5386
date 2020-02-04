# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:31:08 2020

@author: funrr
"""

"""
 Simple example showing evaluating embedding on similarity datasets
"""
#import logging
from six import iteritems
from web.datasets.similarity import fetch_MTurk, fetch_MEN, fetch_WS353, fetch_RG65, fetch_RW, fetch_SimLex999, fetch_TR9856
import numpy as np
from web.embeddings import fetch_CBOW, fetch_GloVe, fetch_SG_GoogleNews, fetch_PDC, fetch_HDC, fetch_LexVec, fetch_conceptnet_numberbatch, fetch_FastText
from web.evaluate import evaluate_similarity


# Configure logging
#logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch GloVe embedding (warning: it might take few minutes)

models = {
'glove': 'fetch_GloVe(corpus="wiki-6B", dim=300)',
'pdc': 'fetch_PDC()',
'hdc':'fetch_HDC()',
'lexvec': 'fetch_LexVec(which="wikipedia+newscrawl-W")',
'cn': 'fetch_conceptnet_numberbatch()',
'w2c': 'fetch_SG_GoogleNews()',
'fastText' : 'fetch_FastText()'
'CBOW': 'fetch_CBOW()'
}


#w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)
#w_PDC = fetch_PDC()
#w_HDC = fetch_HDC()
#w_LexVec = fetch_LexVec(which="wikipedia+newscrawl-W")
#w_conceptnet_numberbatch = fetch_conceptnet_numberbatch()
#w_w2v = fetch_SG_GoogleNews()
#w_fastText = fetch_FastText()  #load_embedding(path, format='word2vec', normalize=True, lower=False, clean_words=False)


# Define tasks
tasks = {
    "MTurk": fetch_MTurk(),
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "RG65":fetch_RG65(),
    "RW":fetch_RW(),
    "SIMLEX999": fetch_SimLex999(),
    "TR9856":fetch_TR9856()
}

result =   np.zeros((7,7))
# Print sample data
#for name, data in iteritems(tasks):
#    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))

# Calculate results using helper function
i = 0
for m_name, m_fun in iteritems(models):   
    j = 0
    try:
        model = eval(m_fun)
    except: 
        i += i
        continue
    for name, data in iteritems(tasks):
        eval_result = evaluate_similarity(model, data.X, data.y)
        print("Spearman correlation of scores on {} {}".format(name, eval_result))
        result[i][j] = eval_result
        j +=1
