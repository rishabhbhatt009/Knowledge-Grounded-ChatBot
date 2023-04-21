import datasets
from sentence_transformers import InputExample
from tqdm.auto import tqdm
from sentence_transformers import datasets as datasets_dup
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses

from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

### Loading the dataset
squad_dev = datasets.load_dataset('squad_v2', split='validation')
print('-'*50, 'Loading Dataset', '-'*50, sep='\n')
print(squad_dev[0])



### Loading Model 
print('-'*50, 'Loading Model', '-'*50, sep='\n')
model = SentenceTransformer('/home/rishabh/GitHub/Knowledge-Graph/model-checkpoint')



### Pre-Processing Data
print('-'*50, 'Processing Dataset', '-'*50, sep='\n')
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
# print(squad_dev[0])



### Converting to dataframe
df = pd.DataFrame(squad_dev)
print(df.head(10))
# df.to_csv('dataframe.csv', index=False)


### Encoding Contexts
# compute the embeddings for each context using the model

print('-'*50, 'Encoding Dataset', '-'*50, sep='\n')
embeddings = np.array([model.encode([c]).squeeze() for c in df['context']])
df['embedding'] = list(embeddings)
print(df.head(10))



# compute the cosine similarity between each embedding and the query
def querydb(query, k=5):
    query_embedding = np.array(model.encode([query]).squeeze())

    df['cosine_sim'] = df['embedding'].apply(lambda x: 1 - cosine(x, query_embedding))
    # sort the DataFrame by cosine similarity and return the top k results
    result = df.sort_values('cosine_sim', ascending=False).head(k)
    
    return result[['id', 'title', 'context']]


### Querying Model
print('-'*50, 'Query - Model', '-'*50, sep='\n')


while True:
    query = input('Enter query : ')
    if query.upper() in ['END', 'EXIT', 'BYE']:
        break
    res = querydb(query, k=5)    
    print(res)
    print('-'*50)
    
	
    


