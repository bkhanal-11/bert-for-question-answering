import torch
from torchsummary import summary
import unittest
from bert import BERT

class TestBERT(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 32
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_head = 12
        
        self.intermediate_size = 4 * self.hidden_size
        self.dropout = 0.1
        self.max_positional_emnddings = 512
        self.layer_norm_eps = 1e-12
        self.mask_idx = 0
        
        self.model = BERT(
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_head,
            d_model=self.hidden_size,
            fully_connected_dim=self.intermediate_size,
            input_vocab_size=self.vocab_size,
            maximum_position_encoding=self.max_positional_emnddings,
            dropout_rate=self.dropout,
            layernorm_eps=self.layer_norm_eps
        )

        self.input_seq_len = 100
        
        self.x = torch.randint(1, self.vocab_size, (self.batch_size, self.input_seq_len))
        self.x_segment = torch.randint(0, 2, (self.batch_size, self.input_seq_len))
        self.mask = torch.ones_like(self.x).bool()
        
    def test_shape(self):
        # Print the total number of parameters in the model
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")    

        # Print a summary of the model architecture and number of parameters
        # summary(self.model, [(self.batch_size, self.input_seq_len), (self.batch_size, self.input_seq_len)], device='cpu')

        output = self.model(self.x, self.x_segment)
        self.assertEqual(output.shape, (self.batch_size, self.input_seq_len, self.hidden_size))
        
    def test_encoder_layers(self):
        for i in range(self.model.num_layers):
            layer = self.model.enc_layers[i]
            
            x = torch.randn((self.batch_size, self.input_seq_len, self.hidden_size))
            output = layer(x)
            
            self.assertEqual(output.shape, (self.batch_size, self.input_seq_len, self.hidden_size))
    
if __name__ == '__main__':
    unittest.main()
