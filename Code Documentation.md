# Code Documentation 

#### 1. Data Collection and Pre-Processing 
- download dataset 
- fetching question-context pairs 
- remove duplicates 
#### 2. Fine Tuning Retriever Model 
- model : microsoft/mpnet-base 
- model pipeline : \[ model , pooler \]


#### 3. Evaluation 
- val squad_v2 dataset 
- generate unique ids for questions and context
- **Note** : idx = questions_ids and idy = context_ids    
- create 3 dicts :
- ir_queries : queries -> question_ids
- ir_corpus : context -> context_ids
- ir_relevant_docs : question_ids -> context_ids
- **Note** : multiple questions can be mapped to single context