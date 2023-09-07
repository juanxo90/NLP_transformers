import csv
from collections import defaultdict
import random
import os
from sentence_transformers import util

import pandas as pd

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_data/')
'''
Dataset preparation to fine tuning retail data
This script is an adaptation from the quora data creation. Please see https://www.sbert.net/examples/training/quora_duplicate_questions/README.html
'''

if not os.path.exists(dataset_path):
    util.http_get(url='https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/tableA.csv',
                  path=os.path.join(dataset_path,'tableA.csv'))
    util.http_get(url='https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/tableB.csv',
                  path=os.path.join(dataset_path,'tableB.csv'))
    util.http_get(url='https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/test.csv',
                  path=os.path.join(dataset_path,'test.csv'))
    util.http_get(url='https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/train.csv',
                  path=os.path.join(dataset_path,'train.csv'))
    util.http_get(url='https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data/valid.csv',
                  path=os.path.join(dataset_path,'valid.csv'))
    
# Getting de data for the correct format
# Getting de data for the correct format
tableA = pd.read_csv(os.path.join(dataset_path,'tableA.csv'))
tableA['id'] = 'a_' + tableA.id.astype(str).str.zfill(4)
tableB = pd.read_csv(os.path.join(dataset_path,'tableB.csv'))
tableB['id'] = 'b_' + tableB.id.astype(str).str.zfill(4)
train = pd.read_csv(os.path.join(dataset_path,'train.csv'))
train['ltable_id'] = 'a_' + train.ltable_id.astype(str).str.zfill(4)
train['rtable_id'] = 'b_' + train.rtable_id.astype(str).str.zfill(4)
test = pd.read_csv(os.path.join(dataset_path,'test.csv'))
test['ltable_id'] = 'a_' + test.ltable_id.astype(str).str.zfill(4)
test['rtable_id'] = 'b_' + test.rtable_id.astype(str).str.zfill(4)
valid = pd.read_csv(os.path.join(dataset_path,'valid.csv'))
valid['ltable_id'] = 'a_' + valid.ltable_id.astype(str).str.zfill(4)
valid['rtable_id'] = 'b_' + valid.rtable_id.astype(str).str.zfill(4)

all_data = (pd.concat([train, test,valid], ignore_index=True)
            .merge(tableA, left_on='ltable_id', right_on='id')
            .merge(tableB, left_on='rtable_id', right_on='id')
           )[['id_x', 'id_y', 'title_x', 'title_y', 'label']]

# creating output directories
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_train'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_train/graph'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_train/information-retrieval'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_train/classification'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_train/duplicate-mining'), exist_ok=True)

sentences = {}
duplicates = defaultdict(lambda: defaultdict(bool))
rows = []

for index, row in all_data.iterrows():
    id1 = row['id_x']
    id2 = row['id_y']
    product1 = row['title_x'].replace("\r", "").replace("\n", " ").replace("\t", " ")
    product2 = row['title_y'].replace("\r", "").replace("\n", " ").replace("\t", " ")
    is_duplicate = str(row['label'])

    if product1 == "" or product2 == "":
        continue

    sentences[id1] = product1
    sentences[id2] = product2
    
    rows.append({'qid1': id1, 'qid2': id2, 'product1': product1, 
             'product2': product2, 'is_duplicate': is_duplicate})
    
    if is_duplicate == '1':
        duplicates[id1][id2] = True
        duplicates[id2][id1] = True
        
#Add transitive closure (if a,b and b,c duplicates => a,c are duplicates)
new_entries = True
while new_entries:
    print("Add transitive closure")
    new_entries = False
    for a in sentences:
        for b in list(duplicates[a]):
            for c in list(duplicates[b]):
                if a != c and not duplicates[a][c]:
                    new_entries = True
                    duplicates[a][c] = True
                    duplicates[c][a] = True
                    
                    
#Distribute rows to train/dev/test split
#Ensure that sets contain distinct sentences
is_assigned = set()
random.shuffle(rows)

train_ids = set()
dev_ids = set()
test_ids = set()

counter = 0
for row in rows:
    if row['qid1'] in is_assigned and row['qid2'] in is_assigned:
        continue
    elif row['qid1'] in is_assigned or row['qid2'] in is_assigned:

        if row['qid2'] in is_assigned: #Ensure that qid1 is assigned and qid2 not yet
            row['qid1'], row['qid2'] = row['qid2'], row['qid1']

        #Move qid2 to the same split as qid1
        target_set = train_ids
        if row['qid1'] in dev_ids:
            target_set = dev_ids
        elif row['qid1'] in test_ids:
            target_set = test_ids

    else:
        #Distribution about 85%/5%/10%
        target_set = train_ids
        if counter%10 == 0:
            target_set = dev_ids
        elif counter%10 == 1 or counter%10 == 2:
            target_set = test_ids
        counter += 1

    #Get the sentence with all duplicates and add it to the respective sets
    target_set.add(row['qid1'])
    is_assigned.add(row['qid1'])

    target_set.add(row['qid2'])
    is_assigned.add(row['qid2'])

    for b in list(duplicates[row['qid1']])+list(duplicates[row['qid2']]):
        target_set.add(b)
        is_assigned.add(b)


#Assert all sets are mutually exclusive
assert len(train_ids.intersection(dev_ids)) == 0
assert len(train_ids.intersection(test_ids)) == 0
assert len(test_ids.intersection(dev_ids)) == 0

print("\nTrain sentences:", len(train_ids))
print("Dev sentences:", len(dev_ids))
print("Test sentences:", len(test_ids))

#Extract the ids for duplicate products for train/dev/test
def get_duplicate_set(ids_set):
    dups_set = set()
    for a in ids_set:
        for b in duplicates[a]:
            ids = sorted([a,b])
            dups_set.add(tuple(ids))
    return dups_set

train_duplicates = get_duplicate_set(train_ids)
dev_duplicates = get_duplicate_set(dev_ids)
test_duplicates = get_duplicate_set(test_ids)


print("\nTrain duplicates", len(train_duplicates))
print("Dev duplicates", len(dev_duplicates))
print("Test duplicates", len(test_duplicates))

####### Output for duplicate mining #######
def write_mining_files(name, ids, dups):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
             'data/retail_train/duplicate-mining/'+name+'_corpus.tsv'), 
              'w', encoding='utf8') as fOut:
        fOut.write("qid\tproduct\n")
        for id in ids:
            fOut.write("{}\t{}\n".format(id, sentences[id]))

    with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
             'data/retail_train/duplicate-mining/'+name+'_duplicates.tsv'), 
              'w', encoding='utf8') as fOut:
        fOut.write("qid1\tqid2\n")
        for a, b in dups:
            fOut.write("{}\t{}\n".format(a, b))

write_mining_files('train', train_ids, train_duplicates)
write_mining_files('dev', dev_ids, dev_duplicates)
write_mining_files('test', test_ids, test_duplicates)

###### Classification dataset #####
with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                       'data/retail_train/classification/train_pairs.tsv'),
          'w', encoding='utf8') as fOutTrain, \
     open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                       'data/retail_train/classification/dev_pairs.tsv'), 
     'w', encoding='utf8') as fOutDev, \
     open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                       'data/retail_train/classification/test_pairs.tsv'),
     'w', encoding='utf8') as fOutTest:
    fOutTrain.write("\t".join(['qid1', 'qid2', 'product1', 'product2', 'is_duplicate'])+"\n")
    fOutDev.write("\t".join(['qid1', 'qid2', 'product1', 'product2', 'is_duplicate']) + "\n")
    fOutTest.write("\t".join(['qid1', 'qid2', 'product1', 'product2', 'is_duplicate']) + "\n")

    for row in rows:
        id1 = row['qid1']
        id2 = row['qid2']

        target = None
        if id1 in train_ids and id2 in train_ids:
            target = fOutTrain
        elif id1 in dev_ids and id2 in dev_ids:
            target = fOutDev
        elif id1 in test_ids and id2 in test_ids:
            target = fOutTest

        if target is not None:
            target.write("\t".join([row['qid1'], 
                                    row['qid2'], 
                                    sentences[id1], 
                                    sentences[id2], 
                                    row['is_duplicate']]))
            target.write("\n")
            
####### Write files for Information Retrieval #####
num_dev_queries = 10_000#5000
num_test_queries = 20_000#10000

corpus_ids = train_ids.copy()
dev_queries = set()
test_queries = set()

#Create dev queries
rnd_dev_ids = sorted(list(dev_ids))
random.shuffle(rnd_dev_ids)

for a in rnd_dev_ids:
    if a not in corpus_ids:
        if len(dev_queries) < num_dev_queries and len(duplicates[a]) > 0:
            dev_queries.add(a)
        else:
            corpus_ids.add(a)

        for b in duplicates[a]:
            if b not in dev_queries:
                corpus_ids.add(b)
                
#Create test queries
rnd_test_ids = sorted(list(test_ids))
random.shuffle(rnd_test_ids)

for a in rnd_test_ids:
    if a not in corpus_ids:
        if len(test_queries) < num_test_queries and len(duplicates[a]) > 0:
            test_queries.add(a)
        else:
            corpus_ids.add(a)

        for b in duplicates[a]:
            if b not in test_queries:
                corpus_ids.add(b)
                

#Write output for information-retrieval
print("\nInformation Retrival Setup")
print("Corpus size:", len(corpus_ids))
print("Dev queries:", len(dev_queries))
print("Test queries:", len(test_queries))

with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                       'data/retail_train/information-retrieval/corpus.tsv'),
          'w', encoding='utf8') as fOut:
    fOut.write("qid\tproduct\n")
    for id in sorted(corpus_ids):
        fOut.write("{}\t{}\n".format(id, sentences[id]))

with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                       'data/retail_train/information-retrieval/dev-queries.tsv'),
          'w', encoding='utf8') as fOut:
    fOut.write("qid\tproduct\tduplicate_qids\n")
    for id in sorted(dev_queries):
        fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))

with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                       'data/retail_train/information-retrieval/test-queries.tsv'),
          'w', encoding='utf8') as fOut:
    fOut.write("qid\tproduct\tduplicate_qids\n")
    for id in sorted(test_queries):
        fOut.write("{}\t{}\t{}\n".format(id, sentences[id], ",".join(duplicates[id])))


print("--DONE--")
