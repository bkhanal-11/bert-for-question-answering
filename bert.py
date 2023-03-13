import torch
import torch.nn as nn
import time

from utils import JointBERTEmbedding, FullyConnected, LayerNorm
from mha import MultiHeadAttention

class BERTLayers(nn.Module):
    """
    The bert encoder layer is composed of a multi-head self-attention mechanism,
    followed by a simple, position-wise fully connected feed-forward network. 
    This architecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, num_heads, d_model, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(BERTLayers, self).__init__()

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      d_model=d_model,
                                      dropout=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=d_model,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, d_model)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, d_model)
        """
        # calculate Self-Attention using Multi-Head Attention
        mha_output = self.mha(x, x, x, mask)  # Self attention (batch_size, input_seq_len, d_model)

        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the output of the multi-head attention layer
        skip_x_attention = self.layernorm1(x + mha_output)

        # pass the output of the multi-head attention layer through a ffn
        ffn_output = self.ffn(skip_x_attention)

        # apply dropout layer to ffn output during training
        ffn_output = self.dropout_ffn(ffn_output)

        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output to get the output of the encoder layer
        encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)

        return encoder_layer_out

class BERT(nn.Module):
    """
    This BERT encoder is composed by a stack of identical layers (EncoderLayers).
    """
    def __init__(self, num_layers, num_heads, d_model, fully_connected_dim,
                 input_vocab_size, maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(BERT, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
    
        self.embedding = JointBERTEmbedding(input_vocab_size, d_model, maximum_position_encoding)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.enc_layers = nn.ModuleList([BERTLayers(num_heads, d_model, fully_connected_dim,
                                                       dropout_rate=dropout_rate, layernorm_eps=layernorm_eps)
                                          for _ in range(num_layers)])
        
    def forward(self, x, x_segment, mask=None):
        """
        Forward pass for the Encoder
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len)
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        seq_len = x.size(1)
        
        # Add position encoding to the input
        x = self.embedding(x, x_segment)
        
        # Apply dropout to the input
        x = self.dropout(x)
        
        # Pass the input through each encoder layer
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        
        encoder_out = x
        
        return encoder_out

if __name__ == '__main__':
    # BERT-small parameters (from paper)
    vocab_size = 30522
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_head = 12
    
    intermediate_size = 4 * hidden_size
    dropout = 0.1
    max_positional_emnddings = 512
    layer_norm_eps = 1e-12
    mask_idx = 0

    x = torch.randint(1, 100, (32, 100))
    x_segment = torch.randint(0, 2, (32, 100))

    model = BERT(
        num_layers=num_hidden_layers,
        num_heads=num_attention_head,
        d_model=hidden_size,
        fully_connected_dim=intermediate_size,
        input_vocab_size=vocab_size,
        maximum_position_encoding=max_positional_emnddings,
        dropout_rate=dropout,
        layernorm_eps=layer_norm_eps
    )

    mask = torch.randint(0, 2, (32, 100))
    mask = (mask != 0).unsqueeze(1).unsqueeze(2)
    start = time.time()
    y = model(x, x_segment)

    
    print(f'INFERENCE TIME = {time.time() - start}sec')
    x = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'NUMBER OF PARAMETERS ARE = {x}')