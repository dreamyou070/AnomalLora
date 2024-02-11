import torch.nn as nn
import torch
import einops
class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return x + pe

class PE_Pooling(nn.Module):
    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()

        #self.pooling_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling_layer = nn.AvgPool2d(kernel_size=(2, 2))
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

        # self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.pooler = ViTPooler(config) if add_pooling_layer else None

    def forward(self, x: torch.Tensor):

        res = int(x.shape[1] ** 0.5)
        # [1] position embedding
        pe = self.positional_encodings[:x.shape[0]]
        x = x + pe

        # [2] pooling
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)  # B,C,H*W
        x = self.pooling_layer(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C

        return x

