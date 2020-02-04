# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:05:56 2020

@author: funrr
"""


"""
 Simple example showing answering analogy questions
"""
import logging
import pandas as pd
from six import iteritems
from web.datasets.analogy import fetch_wordrep, fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2
from web.embeddings import fetch_CBOW, fetch_GloVe, fetch_SG_GoogleNews, fetch_PDC, fetch_HDC, fetch_LexVec, fetch_conceptnet_numberbatch, fetch_FastText
from web.evaluate import evaluate_on_all, evaluate_on_semeval_2012_2, evaluate_on_WordRep
# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch skip-gram trained on GoogleNews corpus and clean it slightly
#w = fetch_SG_GoogleNews(lower=True, clean_words=True)
#w = fetch_GloVe()

# Fetch analogy dataset
#data = fetch_wordrep()

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

tasks = {   
    "all": 'evaluate_on_all(model)',
    "WordRep": 'evaluate_on_WordRep(model, max_pairs = 5)',
}


results = pd.DataFrame(0, index=list(models.keys()), columns=["Google", 'MSR', 'SemEval2012_2', "WordRep" ])



for m_name, m_fun in iteritems(models):   
    try:
        model = eval(m_fun)
    except: 
        print('loading model ' + m_name + " failed")
        continue
    print('loading model ' + m_name + " successful")            
    for name, evaluation in iteritems(tasks):
        try:
            result = eval(evaluation)
        except: 

            print('evaluate ' + name + " failed")
            continue
        print('evaluate ' + name + " successful")
        if name == 'all':
            results.loc[m_name, 'Google'] = result['Google'].values[0]
            results.loc[m_name, 'MSR'] = result['MSR'].values[0]
            results.loc[m_name, 'SemEval2012_2'] = result['SemEval2012_2'].values[0]
        elif name == 'WordRep':
            results.loc[m_name, 'WordRep'] = result.loc['all','accuracy']
            

 


 


#res = evaluate_on_WordRep(w, max_pairs = 50)

#res3 =evaluate_on_all(w)

#WordRep.loc['all','accuracy']
#allre['Google'].values[0]
#allre['MSR'].values[0]
#allre['SemEval2012_2'].values[0]
#for cat in (set(data.category)):
#    print(cat)

# Pick a sample of data and calculate answers
#subset = [50, 1000, 4000, 10000, 14000]
#for id in subset:
#    w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
 #   print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
  #  print("Answer: " + data.y[id])
   # print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))