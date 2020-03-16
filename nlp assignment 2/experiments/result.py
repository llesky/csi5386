import pandas as pd
import sklearn.metrics
import scipy
#merge two file
# results1 = pd.read_csv('./Data/test_results.tsv',sep='\t')
# results2 = pd.read_csv('./Data/test_results2.tsv',sep='\t')

# df = pd.DataFrame({'pair_ID': results1["pair_ID"],
#                    'relatedness_score': results2["relatedness_score"],
#                    'entailment_judgment': results1["entailment_judgement"]})

# df.to_csv('./Data/Result.txt', index=False, sep='\t', encoding='utf-8')  
labels = pd.read_csv('./Data/test_annotated.txt',sep='\t')
results = pd.read_csv('./Data/Result.txt',sep='\t')

confusion_matrix = sklearn.metrics.confusion_matrix(y_true=labels["entailment_judgment"],y_pred=results["entailment_judgment"],labels=["ENTAILMENT", "NEUTRAL", "CONTRADICTION"])
print(confusion_matrix)
correct = confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2]
total = sum(sum(confusion_matrix))
ratio = correct/total
print("accurcy for task 1:",ratio)

#label_num = [0 if i=="ENTAILMENT" else i=="NEUTRAL" for i in labels['relatedness_score']]
#print(label_num)
precision_score = sklearn.metrics.precision_score(labels['entailment_judgment'],results['entailment_judgment'],average=None)
print("precision_score:",precision_score)

recall_score = sklearn.metrics.recall_score(labels['entailment_judgment'],results['entailment_judgment'],average=None)
print("recall_score:",recall_score)

f1_score = sklearn.metrics.f1_score(labels['entailment_judgment'],results['entailment_judgment'],average=None)
print("recall_score:",f1_score)

'''
pearsonr = scipy.stats.pearsonr(labels['relatedness_score'],results['relatedness_score'])[0]
print('Pearson correlation: ' + str(pearsonr))

mse = sklearn.metrics.mean_squared_error(y_true=labels['relatedness_score'],y_pred=results['relatedness_score'])
print('Mean squre error: '+str(mse))

spearmanr = scipy.stats.spearmanr(labels['relatedness_score'],results['relatedness_score'])
print(str(spearmanr))'''


