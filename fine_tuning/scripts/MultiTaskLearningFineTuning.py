import torch
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import csv
import os
import random

### Multi Task Learning fine tuning for retail data, this is an 
### adaptation from the quora question Fine Tuning, 
### for more detail see: https://www.sbert.net/examples/training/quora_duplicate_questions/README.html

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch_device = 'cpu'

model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli',
                           device=torch_device)
num_epochs = 10
train_batch_size = 8

#As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

#Negative pairs should have a distance of at least 0.5
margin = 0.70

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                            'data/retail_train/')

model_save_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 
                               'data/output/training_retailMultiTask-')\
+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

######### Read train data  ##########
train_samples_MultipleNegativesRankingLoss = []
train_samples_ConstrativeLoss = []

with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        train_samples_ConstrativeLoss.append(InputExample(texts=[row['product1'], 
                                                                 row['product2']], 
                                                          label=int(row['is_duplicate'])))
        if row['is_duplicate'] == '1':
            train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[row['product1'], 
                                                                                  row['product2']], 
                                                                           label=1))
            # if A is a duplicate of B, then B is a duplicate of A
            train_samples_MultipleNegativesRankingLoss.append(InputExample(texts=[row['product2'], 
                                                                                  row['product1']], 
                                                                           label=1))  

# Create data loader and loss for MultipleNegativesRankingLoss
train_dataloader_MultipleNegativesRankingLoss = DataLoader(train_samples_MultipleNegativesRankingLoss, 
                                                           shuffle=True, 
                                                           batch_size=train_batch_size)
train_loss_MultipleNegativesRankingLoss = losses.MultipleNegativesRankingLoss(model)


# Create data loader and loss for OnlineContrastiveLoss
train_dataloader_ConstrativeLoss = DataLoader(train_samples_ConstrativeLoss, 
                                              shuffle=True, 
                                              batch_size=train_batch_size)
train_loss_ConstrativeLoss = losses.OnlineContrastiveLoss(model=model, 
                                                          distance_metric=distance_metric, 
                                                          margin=margin)

################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Products pair classification,
# Duplicate Products Mining, and Duplicate Products Information Retrieval
evaluators = []

###### Classification ######
# Given (product1, product2), is this a duplicate or not?
# The evaluator will compute the embeddings for both products and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
with open(os.path.join(dataset_path, "classification/dev_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences1.append(row['product1'])
        dev_sentences2.append(row['product2'])
        dev_labels.append(int(row['is_duplicate']))


binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, 
                                                                dev_sentences2, 
                                                                dev_labels)
evaluators.append(binary_acc_evaluator)

###### Duplicate Products Mining ######
# Given a large corpus of products, identify all duplicates in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_dev_samples = 100_000
dev_sentences = {}
dev_duplicates = []
with open(os.path.join(dataset_path, "duplicate-mining/dev_corpus.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences[row['qid']] = row['product']

        if len(dev_sentences) >= max_dev_samples:
            break

with open(os.path.join(dataset_path, "duplicate-mining/dev_duplicates.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['qid1'] in dev_sentences and row['qid2'] in dev_sentences:
            dev_duplicates.append([row['qid1'], row['qid2']])


# The ParaphraseMiningEvaluator computes the cosine similarity between all sentences and
# extracts a list with the pairs that have the highest similarity. Given the duplicate
# information in dev_duplicates, it then computes and F1 score how well our duplicate mining worked
paraphrase_mining_evaluator = evaluation.ParaphraseMiningEvaluator(dev_sentences, 
                                                                   dev_duplicates, 
                                                                   name='dev')
evaluators.append(paraphrase_mining_evaluator)

###### Duplicate Products Information Retrieval ######
# Given a product and a large corpus of thousands products, 
# find the most relevant (i.e. duplicate) product in that corpus.

# For faster processing, we limit the development corpus to only 10,000 sentences.
max_corpus_size = 100_000

ir_queries = {}             #Our queries (qid => question)
ir_needed_qids = set()      #QIDs we need in the corpus
ir_corpus = {}              #Our corpus (qid => question)
ir_relevant_docs = {}       #Mapping of relevant documents for a given query 
                            #(qid => set([relevant_question_ids])

with open(os.path.join(dataset_path, 'information-retrieval/dev-queries.tsv'), encoding='utf8') as fIn:
    next(fIn) #Skip header
    for line in fIn:
        qid, query, duplicate_ids = line.strip().split('\t')
        duplicate_ids = duplicate_ids.split(',')
        ir_queries[qid] = query
        ir_relevant_docs[qid] = set(duplicate_ids)

        for qid in duplicate_ids:
            ir_needed_qids.add(qid)

# First get all needed relevant documents 
# (i.e., we must ensure, that the relevant products are actually in the corpus
distraction_products = {}
with open(os.path.join(dataset_path, 'information-retrieval/corpus.tsv'), encoding='utf8') as fIn:
    next(fIn) #Skip header
    for line in fIn:
        qid, product = line.strip().split('\t')
        if qid in ir_needed_qids:
            ir_corpus[qid] = product
        else:
            distraction_products[qid] = product

# Now, also add some irrelevant products to fill our corpus
other_qid_list = list(distraction_products.keys())
random.shuffle(other_qid_list)

for qid in other_qid_list[0:max(0, max_corpus_size-len(ir_corpus))]:
    ir_corpus[qid] = distraction_products[qid]
    
#Given queries, a corpus and a mapping with relevant documents, 
# the InformationRetrievalEvaluator computes different IR
# metrices. For our use case MRR@k and Accuracy@k are relevant.
ir_evaluator = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs)

evaluators.append(ir_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three 
# evaluators in a sequential order. We optimize the model with respect
# to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, 
                                               main_score_function=lambda scores: scores[-1])


logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

# Train the model
model.fit(train_objectives=[(train_dataloader_MultipleNegativesRankingLoss, 
                             train_loss_MultipleNegativesRankingLoss), 
                            (train_dataloader_ConstrativeLoss, 
                             train_loss_ConstrativeLoss)],
          evaluator=seq_evaluator,
          epochs=num_epochs,
          warmup_steps=1000,
          output_path=model_save_path
          )
