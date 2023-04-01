from sentence_transformers import SentenceTransformer

print('-'*50, 'Loading Model', '-'*50, sep='\n')
path = '/home/rishabh/GitHub/Knowledge-Graph/model-checkpoint'
model = SentenceTransformer(path)

print('Model Loaded from',path)

print('-'*50, 'Query - Model', '-'*50, sep='\n')
query = 'Start'
while query != 'end':
	if query != 'Start':
		response = model.encode(query)
		print(response)
	query = input('Enter query : ')

print('')
print('Thnak you')

