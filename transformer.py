import torch.nn as nn


class TransformerRationalePredictor(nn.Module):
  def __init__(self, num_layers, d_model, num_heads,
               dff, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.linear = nn.Linear(self.d_model, self.d_model*num_heads)

    self.enc_layers = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=self.d_model*num_heads,
                  nhead=num_heads,
                  dim_feedforward=dff,
                  dropout=dropout_rate,
                  batch_first=True)
        for _ in range(num_layers)])
    
    self.linear2 = nn.Linear(self.d_model*num_heads, self.d_model)
        
  
  def forward(self, x):
    '''
    inputs: 
            x : [batch_size, num_tokens, d_model]
    '''
    x = self.linear (x)
    x = self.enc_layers(x)
    x = self.linear2(x)
    return x  # Shape `(batch_size, seq_len, d_model)`
