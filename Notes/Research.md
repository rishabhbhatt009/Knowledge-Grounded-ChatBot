	## Research

1. How do I ground large language models ? 
	- using a large corpus of annotated data
	- sentiment analysis and entity recognition
	- additional contextual information (geographic information, domain-specific knowledge, and other structured data)
2. Methods to ground -
	1. **Pretraining on large-scale datasets**: Pretraining on large-scale datasets such as ImageNet or Common Crawl can provide the model with a diverse range of real-world experiences.
	2. **Fine-tuning on task-specific datasets**: Fine-tuning on datasets specifically designed for the task of interest, such as natural language understanding, can help the model better understand the context and nuances of the task.
	3. **Multi-modal learning**: Incorporating information from multiple modalities, such as images, text, and audio, can provide the model with a more comprehensive understanding of the real-world.
	4. **Active learning**: An active learning approach, where the model selects the examples it wants to learn from, can help the model focus on the most important examples for its task.
	5. **Knowledge distillation**: Transferring knowledge from a smaller, more specialized model to a larger one can help the larger model retain the specialized knowledge while leveraging its larger capacity to perform better on other tasks.
3. Combining pre-trained language models with structured knowledge :
	1. **Fine-tuning pre-trained language models**: One can fine-tune a pre-trained language model on a task-specific dataset that contains structured knowledge to make it more proficient in utilizing structured knowledge.
	2. **Knowledge distillation**: In this method, one can train a smaller model (student model) to mimic the behavior of a larger pre-trained language model (teacher model) on structured knowledge tasks.
	3. **Knowledge graph-augmented models**: A knowledge graph can be used to augment a pre-trained language model by incorporating structured knowledge in the form of entities and relationships. The model can then be fine-tuned on a task-specific dataset to better utilize the structured knowledge.
	4. **Hybrid models**: One can also develop hybrid models that combine the strengths of pre-trained language models with hand-crafted rule-based systems or symbolic reasoning systems to better utilize structured knowledge.   
	   
	Note : The specific implementation that works best will depend on the task at hand and the nature of the structured knowledge being utilized.

Papers :
1. Papers that use annotated data to ground language models:
	1.  "ImageBERT: Pretraining ResNet-50 for Image-Text Matching" (CVPR 2019) - This paper presents a pretraining method for large language models that leverages annotated image-text pairs to learn cross-modal representations for image and text. [https://arxiv.org/abs/1909.13370](https://arxiv.org/abs/1909.13370)
	2.  "Fine-Tuning Pretrained Language Models with Adversarial Loss" (EMNLP 2019) - This paper proposes using annotated data and adversarial loss to fine-tune large language models for specific tasks such as sentiment analysis and question answering. [https://arxiv.org/abs/1905.01052](https://arxiv.org/abs/1905.01052)
	3.  "Pretraining-Based Natural Language Processing Tasks Using Transformer Models: A Survey" (ArXiv 2021) - This survey paper provides a comprehensive overview of recent work in the area of pretraining large language models using annotated data, including both supervised and unsupervised approaches. [https://arxiv.org/abs/2010.11926](https://arxiv.org/abs/2010.11926)
	4.  "ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations" (ArXiv 2019) - This paper presents a pretraining method for large language models that leverages annotated data to learn self-supervised representations for natural language understanding. [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
	5.  "Task-Agnostic Pretraining for Multi-Task NLP with Differentiable Head Architectures" (ArXiv 2020) - This paper proposes using annotated data to pretrain a large language model that can be fine-tuned to perform multiple NLP tasks. [https://arxiv.org/abs/2002.12412](https://arxiv.org/abs/2002.12412)
2. Papers that implement knowledge graphs with large language models:
	1.  "GraphBERT: A Graph-Structured Pretraining Method for Language Understanding" (ArXiv 2020) - This paper proposes a pretraining method for large language models that leverages knowledge graphs to learn graph-structured representations for natural language understanding. The code for this paper is available on GitHub: [https://github.com/thunlp/GraphBERT](https://github.com/thunlp/GraphBERT)    
	2.  "OpenKG-BERT: Pretraining Language Representations on Large Scale Knowledge Graphs" (ArXiv 2020) - This paper presents a pretraining method for large language models that leverages knowledge graphs to learn representations that are grounded in structured knowledge. The code for this paper is available on GitHub: [https://github.com/thunlp/OpenKG-BERT](https://github.com/thunlp/OpenKG-BERT)
	3.  "Exploiting Knowledge Graphs for Pretraining Large Language Models" (ArXiv 2020) - This paper proposes a knowledge graph-based pretraining method for large language models that leverages both unstructured text and structured knowledge to learn better representations for natural language understanding. The code for this paper is available on GitHub: [https://github.com/thunlp/KGPretraining](https://github.com/thunlp/KGPretraining)
	4.  "ERNIE: Enhanced Language Representation with Informative Entities" (AAAI 2020) - This paper presents a pretraining method for large language models that leverages knowledge graphs to learn entity-aware representations for natural language understanding. The code for this paper is available on GitHub: [https://github.com/PaddlePaddle/ERNIE](https://github.com/PaddlePaddle/ERNIE)
	5.  "Language Models Are Unsupervised Multitask Learners" (ArXiv 2019) - This paper shows how knowledge graphs can be used to train large language models in a multitask learning setting, where the model is trained to perform multiple NLP tasks using a single pretraining process. The code for this paper is available on GitHub: [https://github.com/openai/language-models](https://github.com/openai/language-models)
3. Papers that implement task graphs with large language models:
	1.  "Task-Oriented Graph Reasoning Networks" (ArXiv 2020) - This paper proposes a new approach to using graph-based representations for task-oriented language understanding, using a graph reasoning network that can reason over task graphs to perform a wide range of NLP tasks. [https://arxiv.org/abs/2010.11426](https://arxiv.org/abs/2010.11426)
	2.  "Graph-Structured Representation for Multi-Turn Reasoning in Task-Oriented Dialogue" (ArXiv 2019) - This paper proposes a graph-structured representation for multi-turn reasoning in task-oriented dialogue, using a graph neural network to learn representations for dialogue state and to reason over the graph structure to make predictions. [https://arxiv.org/abs/1909.01310](https://arxiv.org/abs/1909.01310)    
	3.  "Exploring Graph Convolutional Networks for Task-Oriented Dialogue Systems" (ArXiv 2018) - This paper proposes using graph convolutional networks for task-oriented dialogue systems, using a graph representation to capture the dependencies between dialogue turns and to perform multi-turn reasoning over the graph structure. [https://arxiv.org/abs/1807.05015](https://arxiv.org/abs/1807.05015)
	4.  "Joint Task-Completion and Turn-Taking in Multi-Turn Dialogue with Deep Reinforcement Learning" (ArXiv 2016) - This paper presents a deep reinforcement learning approach to joint task-completion and turn-taking in multi-turn dialogue, using a graph representation to capture the dependencies between dialogue turns and to make predictions about the next action in the dialogue. [https://arxiv.org/abs/1610.01269](https://arxiv.org/abs/1610.01269)
	5.  "End-to-End Task-Completion Neural Dialogue Systems" (ArXiv 2016) - This paper presents an end-to-end neural dialogue system for task-completion, using a graph representation to capture the dependencies between dialogue turns and to make predictions about the next action in the dialogue. [https://arxiv.org/abs/1603.01353](https://arxiv.org/abs/1603.01353)

Scraping Data : 
Open-Graph-Protocol-Tags 

Fine Tuning GPT :
- prep data 
- send for fine tune 
- cost ~ %10
- pre train with image links and see results

- Through equipping LMs with a module that retrieves such documents from a database given a context, it is possible to match certain capabilities of some of the largest LMswhile having less parameters (Borgeaud et al., 2022; Izacard et al., 2022). Note that the resulting model is now non-parametric since it can query external data sources.
- Augmented Language Models (ALMs)

---

#### How to build a Q&A AI in Python (Open-domain Question-Answering) - James Briggs | [YouTube](https://www.youtube.com/watch?v=w1dMEWm7jBc&ab_channel=JamesBriggs)
Great resource to 
- **Retrieve Model** = 
- Question -> Retrieve Model -> Token Vector -> Pooling Layer Query Vector 
- **Vector Database** = Context Vectors (Indexed Offline)
- Query Vector -> Vector Database -> top k context 
- **Reader Model**
- Why Fine Tuning Model : Common knowledge area easy to fine, more specific use case require fine tuning 
- **Data** = {Question, Context} pairs - squad_v2
- 