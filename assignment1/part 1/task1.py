import nltk
from nltk.corpus import words
from nltk.corpus import brown
import spacy
import re
nlp = spacy.load("en")

def get_tokens_from_file(filename,line=False):
    doc = open(filename,encoding='utf-8-sig').read()
    doc = doc.split("\n\n\n")
    output = []
    if line == False:
        for i in doc:
            output.extend(i.split("\n"))
    else:
        for i in doc:
            output.append(i.split("\n"))
    return output[:-1]

def write_format_tokenized_top20(tokens):
    output = ""
    i=1
    for token in tokens:
        line_format = ""
        for x in token:
            line_format +="["+x+"] "
        output+= str(i)+" "+line_format+"\n\n"
        if i==20:
            break
        i+=1
    
    print(output)

#section A
tokens = get_tokens_from_file("microblog2011_tokenized.txt",line=True)
write_format_tokenized_top20(tokens)

#section B
tokens = get_tokens_from_file("microblog2011_tokenized.txt")

print("number of tokens:",len(tokens))

fdist = nltk.FreqDist([i.lower() for i in tokens])
print("types fo tokens:",len(fdist.keys()))

print("ratio:",len(fdist.keys())/len(tokens))

#section C
all_tokens_dic = fdist.most_common()

output = ""
for i in range(len(all_tokens_dic)):
    output+=str(i)+" "+str(all_tokens_dic[i])+"\n"
file_tokens = open("Tokens.txt","w",encoding='utf-8-sig')
file_tokens.write(output)
file_tokens.close()


#section D
'''
countD = 0
for x,y in fdist.items():
    if y==1:
        countD+=1
print(countD)'''

#secion E
'''
word_regex = re.compile("^[a-z]+$", re.I)
outputE=""
file_E = open("E.txt","w",encoding='utf-8-sig')
i=1
for token in all_tokens_dic:
    if word_regex.match(token[0]):
        if (i<101):
            print(i,token)
        outputE+=str(i)+"\n"+token[0]+"\n"+str(token[1])+"\n\n\n"
        i+=1

file_E.write(outputE)
file_E.close()

'''
#section F

word_tokens = get_tokens_from_file("E.txt",line=True)
def get_stopwords(filename):
    doc = open(filename,encoding='utf-8-sig').read()
    return doc.split("\n")[:-1]

stop_words = get_stopwords("StopWords.txt")
outputF=""
file_F = open("F.txt","w",encoding='utf-8-sig')
i=1
for token in word_tokens:
    if token[1] not in stop_words:
        if (i<101):
            print(i,(token[1],int(token[2])))
        outputF+=str(i)+"\n"+token[1]+"\n"+str(token[2])+"\n\n\n"
        i+=1
file_F.write(outputF)
file_F.close()


#section G
word_tokens = get_tokens_from_file("F.txt",line=True)
word_tokens = [i[1] for i in word_tokens]


pairs = nltk.bigrams([i.lower() for i in tokens])
pairs_freq=nltk.FreqDist(pairs).most_common()

pair_tokens = []
outputG = ""
'''
'''
index=1
for i in pairs_freq:
    if i[0][0] in word_tokens and i[0][1] in word_tokens:
        print(index,i)
        outputG+=str(index)+"\n"+str(i[0])+"\n"+str(i[1])+"\n\n\n"
        index+=1
      
file_G = open("G.txt","w",encoding='utf-8-sig')
file_G.write(outputG)
file_G.close()