import torch.nn as nn
import torch

class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe