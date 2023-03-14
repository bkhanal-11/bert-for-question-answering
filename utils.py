import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim, fully_connected_dim):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, fully_connected_dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(fully_connected_dim, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        return self.fc2(x)

class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        return self.scale * normalized + self.bias

class JointBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(JointBERTEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, d_model, padding_idx=0)

        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_segment):
        token_embeddings = self.token_embedding(x)
        segment_embeddings = self.segment_embedding(x_segment)

        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x)
        position_embeddings = self.position_embedding(position_ids)

        embeddings = token_embeddings + segment_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings

if __name__ =="__main__":
    model = JointBERTEmbedding(vocab_size=30522, d_model=768, max_seq_len=512)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")