import datasets
from sentence_transformers import InputExample
from tqdm.auto import tqdm
from sentence_transformers import datasets as datasets_dup
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import losses

print('-'*50, 'Importing Dataset', '-'*50, sep='\n')

# importing dataset 
squad = datasets.load_dataset('squad_v2', split='train')

# processing dataset 
train = []
for row in tqdm(squad):
    train.append(InputExample(
        texts=[row['question'], row['context']]
    ))

# removing duplicates 
batch_size = 16

loader = datasets_dup.NoDuplicatesDataLoader(
    train, batch_size=batch_size
)

print('-'*50, 'Initializing Model', '-'*50, sep='\n')

# init model 
bert = models.Transformer('microsoft/mpnet-base')

pooler = models.Pooling(
    bert.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[bert, pooler], device='cuda')
loss = losses.MultipleNegativesRankingLoss(model)

print('-'*50, 'Training Model', '-'*50, sep='\n')

# training 
epochs = 1
warmup_steps = int(len(loader) * epochs * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path='mpnet-mnr-squad2',
    show_progress_bar=True
)

modelPath = '/home/rishabh/GitHub/Knowledge-Graph/model-checkpoint'
model.save(modelPath)
