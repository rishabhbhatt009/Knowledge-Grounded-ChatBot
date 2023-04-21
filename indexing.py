import datasets
from sentence_transformers import InputExample
from tqdm.auto import tqdm
from sentence_transformers import datasets as datasets_dup
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses

from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
from scipy.spatial.distance import cosine

### Loading the dataset
squad_dev = datasets.load_dataset('squad_v2', split='validation')
print('-'*50, 'Loading Dataset', '-'*50, sep='\n')
print(squad_dev[0])

### Loading Model 
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


### Encoding Contexts
print('-'*50, 'Encoding Dataset', '-'*50, sep='\n')
squad_dev = squad_dev.map(lambda x: {
    'encoding': model.encode(x['context']).tolist()
}, batched=True, batch_size=4)
print(squad_dev)
# print(squad_dev[0])


### Converting to dataframe
df = pd.DataFrame(squad_dev)
print(df.head(10))
df.to_csv('dataframe.csv', index=False)

### Fetching Context
def calculate_cosine_distance(vector1, vector2):
    	return cosine(vector1, vector2)

def fetch_context(df, query_vector, k):
	df_copy = df.copy()
	df_copy['cosine_distance'] = df_copy['encoding'].apply(lambda x: calculate_cosine_distance(query_vector, x))
	df_copy.sort_values('cosine_distance', ascending=False, inplace=True)

	print('Best Context Match:')
	#for index, row in df.head(k).iterrows():
	#   for col, value in row.items():
        #	print(col, ":", value)
	#    print()

	print(df_copy.head(k))

### Querying Model
print('-'*50, 'Query - Model', '-'*50, sep='\n')
query = 'Start'
while query != 'end':
	if query != 'Start':
		query_vector = model.encode(query)
		qv = pd.DataFrame(query_vector)
		print(len(qv))
		qv.to_csv('qv.csv', index=False)

		fetch_context(df, query_vector, 5)
	query = input('Enter query : ')

