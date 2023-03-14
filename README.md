# Bidirectional Encoder Represenation from Transformers (BERT)

BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art pre-trained language model developed by Google. It is a deep neural network architecture that has been trained on a large corpus of text data, including the entire Wikipedia and BookCorpus. BERT is able to understand the meaning of words in the context of a sentence, which is a major breakthrough in natural language processing.

BERT is based on the transformer architecture, which is a type of neural network that is particularly effective at processing sequential data, such as text. The transformer model was introduced by Vaswani et al. in their 2017 paper "Attention Is All You Need". The transformer consists of an encoder and a decoder, which are both made up of multiple layers of attention mechanisms and feedforward networks. The encoder is used for tasks such as language modeling, while the decoder is used for tasks such as machine translation.

BERT uses a variant of the transformer architecture called the bidirectional transformer encoder. The bidirectional transformer encoder processes the entire input sequence at once, rather than processing it one token at a time. This allows BERT to capture the context of each word in the sentence, including both the words that come before and after it. BERT is trained using two main objectives: masked language modeling and next sentence prediction.

In **Masked Language Modeling (MLM)**, BERT is given a sentence and some of the words in the sentence are randomly masked. The objective of the model is to predict the masked words based on the context of the sentence. This forces the model to learn representations of each word that are based on the context in which it appears, rather than just its individual meaning.

In **Next Sentence Prediction (NSP)**, BERT is given two sentences and the objective of the model is to predict whether the second sentence is a continuation of the first sentence or not. This task helps BERT to learn the relationship between different sentences and the context in which they appear.

![BERT Model for QA](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.1UuXPxv_1thFXJKSfQT1nQHaDP%26pid%3DApi&f=1&ipt=89a6f6b960ccf2484aee5bdf265525854385d7dbf2e733e2b9d868f2875ef8fc&ipo=images)

In BERT, there are three types of embeddings: token embeddings, segment embeddings, and position embeddings. Each of these embeddings serves a specific purpose in allowing BERT to process text data and understand the relationships between different parts of a sentence.

1. Token embeddings:

Token embeddings are learned representations of each token (or word) in the input sentence. Each token is represented by a fixed-size vector, which is learned during training. This vector represents the meaning of the token in the context of the sentence. The token embeddings are then used as input to the transformer encoder in BERT. This allows the model to understand the meaning of each word in the sentence based on the context in which it appears.

2. Segment embeddings:

Segment embeddings are used to distinguish between two sentences in the input. In some natural language processing tasks, such as question-answering, the input consists of two sentences: a question and an answer. In order for BERT to process these two sentences together, it needs to be able to distinguish between them. This is done using segment embeddings. A segment embedding is a learned representation of each sentence in the input. Each token in the first sentence is assigned a segment embedding of 0, while each token in the second sentence is assigned a segment embedding of 1. This allows the transformer encoder to distinguish between the two sentences and understand the relationships between them.

3. Position embeddings:

Position embeddings are used to encode the position of each token in the input sequence. Unlike recurrent neural networks (RNNs), transformers do not inherently understand the order of tokens in a sequence. Therefore, BERT uses position embeddings to encode the order of tokens in the input. The position embedding of each token is learned during training and is added to the token embedding before being passed to the transformer encoder. This allows the model to understand the relationships between different parts of the sentence based on their position in the sequence.

### Few lookworthy Resources

[1] Number of parameters in BERT: [How is the number of parameters be calculated in BERT model?](https://stackoverflow.com/a/71472362)

[2] Motivation behind different Embeddings in BERT: [How the Embedding Layers in BERT were implemented?](https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a)

[3] Visualization of training for pre-trained BERT: [BERT Summarization](https://github.com/VincentK1991/BERT_summarization_1/blob/master/notebook/Primer_to_BERT_extractive_summarization_March_25_2020.ipynb)