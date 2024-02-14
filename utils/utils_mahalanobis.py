import torch
import torch.nn as nn
def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

def gen_mahal_loss(args, anormal_feat_list, normal_feat_list):
    anormal_feats = torch.cat(anormal_feat_list, dim=0)
    normal_feats = torch.cat(normal_feat_list, dim=0)
    # [2] mu and cov
    mu = torch.mean(normal_feats, dim=0)
    cov = torch.cov(normal_feats.transpose(0, 1))
    # [3] mahalanobis distance
    anormal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in anormal_feats]
    anormal_dist_mean = torch.tensor(anormal_mahalanobis_dists).mean()
    normal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in normal_feats]
    normal_dist_mean = torch.tensor(normal_mahalanobis_dists).mean()
    # [4] loss
    total_dist = normal_dist_mean + anormal_dist_mean
    normal_dist_loss = normal_dist_mean / total_dist
    normal_dist_loss = normal_dist_loss * args.dist_loss_weight
    return normal_dist_loss