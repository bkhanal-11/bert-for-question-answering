import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        # Calculate dot products between the query and the key
        matmul_qk = torch.einsum('bqhd,bkhd->bhqk', [queries, keys])

        dk = keys.size()[-1]
        scaled_attention_logits = matmul_qk / (dk ** 0.5)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask==0, float('-1e20')) 
        
        # Apply softmax function to obtain attention weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Apply dropout to the attention weights
        attention_weights_dropout = self.dropout(attention_weights)

        output = torch.einsum('bhqv,bvhd->bqhd', [attention_weights_dropout, values])

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.self_attention = ScaledDotProductAttention(dropout)
        
        self.query_linear = nn.Linear(self.head_dim, self.head_dim)
        self.key_linear = nn.Linear(self.head_dim, self.head_dim)
        self.value_linear = nn.Linear(self.head_dim, self.head_dim)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformations
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Linear transformations
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Scaled dot-product attention
        context = self.self_attention(query, key, value, mask)
        context = context.reshape(batch_size, -1, self.num_heads*self.head_dim)

        # Concatenate heads and apply final linear transformation
        output = self.out_linear(context)

        return output

if __name__ == "__main__":
    # Define input tensor
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    x = torch.randn(batch_size, seq_len, d_model)

    # Create MultiHeadAttention module
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)

    # Obtain output tensor
    output = mha(x, x, x)

    # Print output tensor shape
    print(output.shape)