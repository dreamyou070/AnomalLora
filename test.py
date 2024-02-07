import torch

recon_query = torch.randn(4, 2)
recon_query = recon_query / (torch.norm(recon_query, dim=1, keepdim=True))
print(recon_query)
result = torch.diag(recon_query @ recon_query.T)
print(result)