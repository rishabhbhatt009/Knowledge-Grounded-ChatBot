### Open-domain Question-Answering-James Briggs
Resource : [YouTube](https://www.youtube.com/watch?v=w1dMEWm7jBc&ab_channel=JamesBriggs)
Article : 

#### 1. Task of Question Answering 
- What happens when we ask questions to google 
- questions -> relevant paragraph -> answer is highlighted 
- traditional search : uses keywords

#### 2. Open-Domain Question Answering (ODQA)
- question answering pipeline
- **Input** : Questions 
- **Retriever Model** :
	- fine-tuned LLM
	- it takes up a query and converts it into a vector
	- it allows us to encode meaning instead of just key words 
- *Question -> Retrieve Model -> Token Vector -> Pooling Layer Query Vector* 
- **Vector Database** : 
	- Context Vectors (Indexed Offline)
	- generated offline using the same model 
- *Query Vector -> Vector Database -> top k context* 
- **Reader Model** : 
	- given the question and the context vectors  generates a precise answer
	- natural language answer  

#### 3. Fine-Tuning Retriever Model
- Why do we need to fine-tune a pre-trained LLM for our retriever model ?
	- Common knowledge v/s specific use case 
	- while its easy to find data and models for common knowledge, its difficult to find them for latter
	- therefore, we need to fine-tune our own models 
- How do we fine-tune a pre-trained LLM: 
	- to train we need *{Question, Context}* pairs
	- we optimize on minimizing distance between similar pairs and maximizing distance between dissimilar pair 
- What data for fine tuning model 
	- squad_v2 dataset : open source dataset with train, validation split 
	- **Data** = {Question, Context} pairs	
	- no labels required in our ? 
		- *since we are going to be training with multiple ranking negative loss* 
	- need to de-duplicate ? 
		- model looks at pairs of questions context and optimizes based on similarity b/w question and its context and the dissimilarity b/w of vector pairs. if we have duplicates in our batches the . 
- Model pipeline :
	- microsoft/mp-net + pooler 
	- sentence -> \[model\] -> token vectors -> \[pooling layer\] -> **sentence vector**
- Model-Training 
	- **Loss** : MNR Loss for training