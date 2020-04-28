from datamanager import DataManager
from collections import Counter
import numpy as np

datamanager = DataManager('a')

for i in datamanager.tweet_train:
    sen = ""
    for x in i:
        sen+=" "+x
    print(sen)

# c = []
# for i in datamanager.tweet_test:
#     for x in i:
#         c.append(x)

# c = Counter(c)
# print(c.most_common(20))
