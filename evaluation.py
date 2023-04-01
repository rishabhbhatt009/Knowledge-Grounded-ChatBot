import datasets
from sentence_transformers import InputExample
from tqdm.auto import tqdm
from sentence_transformers import datasets as datasets_dup
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses

from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd


squad_dev = datasets.load_dataset('squad_v2', split='validation')
print('-'*50, 'Loading Dataset', '-'*50, sep='\n')
print(squad_dev[0])



print('-'*50, 'Creating DataFrame', '-'*50, sep='\n')

squad_df = list()
for row in tqdm(squad_dev):
    squad_df.append({
        'question': row['question'],
        'context': row['context'],
        'id': row['id']
    })
squad_df = pd.DataFrame(squad_df)
print(squad_df.head())


print('-'*50, 'Droping Duplicates', '-'*50, sep='\n')

no_dupe = squad_df.drop_duplicates(
    subset='context',
    keep='first'
)
# also drop question column
no_dupe = no_dupe.drop(columns=['question'])
# and give each context a slightly unique ID
no_dupe['id'] = no_dupe['id'] + 'con'
print(no_dupe.head())
print()

squad_df = pd.merge(squad_df, no_dupe, how='inner', on='context')
print(squad_df.head())

### idx = questions and idy = context


print('-'*50, 'Information Retrieval', '-'*50, sep='\n')
ir_queries = {
    row['id_x']: row['question'] for i, row in squad_df.iterrows()
}

count = 0
for key in ir_queries : 
    print(f'{key} : {ir_queries[key]}')
    count += 1
    if count == 5:
        break 

ir_corpus = {
    row['id_y']: row['context'] for i, row in squad_df.iterrows()
}

count = 0
for key in ir_corpus : 
    print(f'{key} : {ir_corpus[key]}')
    count += 1
    if count == 5:
        break 



print('-'*50, 'IR Document', '-'*50, sep='\n')

ir_relevant_docs = {key: [] for key in squad_df['id_x'].unique()}
for i, row in squad_df.iterrows():
    # we append in the case of a question ID being connected to
    # multiple context IDs
    ir_relevant_docs[row['id_x']].append(row['id_y'])
# this must be in format {question_id: {set of context_ids}}
ir_relevant_docs = {key: set(values) for key, values in ir_relevant_docs.items()}

count = 0
for key in ir_relevant_docs : 
    print(f'{key} : {ir_relevant_docs[key]}')
    count += 1
    if count == 5:
        break 



print('-'*50, 'Evaluation', '-'*50, sep='\n')

ir_eval = InformationRetrievalEvaluator(
    ir_queries, ir_corpus, ir_relevant_docs
)

model = SentenceTransformer('/home/rishabh/GitHub/Knowledge-Graph/model-checkpoint')
score = ir_eval(model)

print(f'Evaluation Score for Model : {score}')
