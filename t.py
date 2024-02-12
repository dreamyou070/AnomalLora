import torch.nn as nn
import torch
import einops


query = torch.randn(1, 4, 64)
query_pos = torch.randn(1, 4, 64)
query = torch.cat([query, query_pos], dim=-1)
print(query.shape) # torch.Size([1, 4, 128])
class PE_Pooling(nn.Module):
    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 4):
        super().__init__()

        #self.pooling_layer = nn.MaxPool2d(kernel_size=(2, 2))
        self.pooling_layer = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1)
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1)
        x = x + pe
        x = self.pooling_layer(x)
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x

x = torch.randn(1,4,64,64)
pe = PE_Pooling()
out = pe(x)
print(out.shape) # torch.Size([1, 320, 32, 32])