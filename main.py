import numpy as np
import pandas as pd
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
raw_data = []
with open("./data/anonymous-msweb.data", 'r') as f:
    for line in f.readlines():
        line = line.split('\n')[0].split(',')
        raw_data.append(line)

web_attr = []
web_dict = {}
vote_info = []
usr_idx = -1
usr_vote = []
vote_info2 = []

for data in raw_data:
    if data[0] == 'A':
        web_attr.append([data[1], data[3]])
        web_dict[data[1]]=data[3]
    if data[0] == 'C':
        if len(usr_vote) > 0:
            vote_info2.append([usr_idx, usr_vote])
            usr_vote = []
        usr_idx = data[2]
    if data[0] == 'V':
        usr_vote.append(web_dict[data[1]])
        vote_info.append([usr_idx, web_dict[data[1]]])
if len(usr_vote) > 0:
    vote_info2.append([usr_idx, usr_vote])

website = pd.DataFrame(web_attr, columns=['idx', 'title'])
website = website.set_index(['idx'])
vote = pd.DataFrame(vote_info, columns=['usr', 'website'])
vote2 = pd.DataFrame(vote_info2, columns=['usr', 'website'])

web_cnt = len(web_attr)
usr_cnt = len(vote_info2)

print(website)
print(vote)
print(vote2)

print(vote['website'].value_counts())
vote['website'].value_counts().plot(kind='bar')
ax = plt.gca()
ax.xaxis.set_major_locator(plt.NullLocator())
plt.show()


data2 = vote2['website'].str.join(',')
data2 = data2.str.get_dummies(',')
frequent_itemsets = apriori(data2,min_support=0.05,use_colnames=True)
frequent_itemsets.sort_values(by='support',ascending=False, inplace=True)
print(frequent_itemsets)
print('='*100)
rules = association_rules(frequent_itemsets, metric='lift',min_threshold=1.5)
rules.sort_values(by='lift',ascending=False, inplace=True)
pd.set_option('display.max_columns',None)
print(rules.loc[:,['antecedents','consequents','support','confidence','lift']].to_string(index=False))


