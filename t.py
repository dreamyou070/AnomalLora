import torch.nn as nn
import torch
import einops


query = torch.randn(4)
query_list = [query, query, query, query, query, query, query, query, query, query, query, query]
query = torch.stack(query_list, dim=0).mean(dim=0)
print(query.shape) # torch.Size([1, 12, 64])
