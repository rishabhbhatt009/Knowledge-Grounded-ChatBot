import datasets
from sentence_transformers import InputExample
from tqdm.auto import tqdm
from sentence_transformers import datasets as datasets_dup
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses

from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd


### Loading the dataset
squad_dev = datasets.load_dataset('squad_v2', split='validation')
print('-'*50, 'Loading Dataset', '-'*50, sep='\n')
print(squad_dev[0])

### Loading Model 
# model = SentenceTransformer('/home/rishabh/GitHub/Knowledge-Graph/model-checkpoint')


### Pre-Processing Data
unique_contexts = []
unique_ids = []

# make list of IDs that represent only first instance of
# each context
for row in squad_dev:
    if row['context'] not in unique_contexts:
        unique_contexts.append(row['context'])
        unique_ids.append(row['id'])

# now filter out any samples that aren't included in unique IDs
squad_dev = squad_dev.filter(lambda x: True if x['id'] in unique_ids else False)
print(squad_dev)


### Encoding Contexts
squad_dev = squad_dev.map(lambda x: {
    'encoding': model.encode(x['context']).tolist()
}, batched=True, batch_size=4)
print(squad_dev)




