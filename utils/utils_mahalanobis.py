import torch
import torch.nn as nn
def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

def gen_mahal_loss(args, anormal_feat_list, normal_feat_list, mu, cov):

    normal_feats = torch.cat(normal_feat_list, dim=0)

    if args.previous_mahal :
        if mu is None:
            mu = torch.mean(normal_feats, dim=0)
            cov = torch.cov(normal_feats.transpose(0, 1))

        if anormal_feat_list is not None:
            anormal_feats = torch.cat(anormal_feat_list, dim=0)
            anormal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in anormal_feats]
            anormal_dist_mean = torch.tensor(anormal_mahalanobis_dists).mean()

        normal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in normal_feats]
        normal_dist_max = torch.tensor(normal_mahalanobis_dists).max()

        # [4] loss
        if anormal_feat_list is not None:
            total_dist = normal_dist_max + anormal_dist_mean
        else :
            total_dist = normal_dist_max
        normal_dist_loss = normal_dist_max / total_dist
        normal_dist_loss = normal_dist_loss * args.dist_loss_weight
        mu = torch.mean(normal_feats, dim=0)
        cov = torch.cov(normal_feats.transpose(0, 1))
    else :
        mu = torch.mean(normal_feats, dim=0)
        cov = torch.cov(normal_feats.transpose(0, 1))
        if anormal_feat_list is not None:
            anormal_feats = torch.cat(anormal_feat_list, dim=0)
            anormal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in anormal_feats]
            anormal_dist_mean = torch.tensor(anormal_mahalanobis_dists).mean()

        normal_mahalanobis_dists = [mahal(feat, mu, cov) for feat in normal_feats]
        normal_dist_max = torch.tensor(normal_mahalanobis_dists).max()

        # [4] loss
        if anormal_feat_list is not None:
            total_dist = normal_dist_max + anormal_dist_mean
        else:
            total_dist = normal_dist_max
        normal_dist_loss = normal_dist_max / total_dist
        normal_dist_loss = normal_dist_loss * args.dist_loss_weight
        mu = None
        cov = None
    return normal_dist_max, normal_dist_loss, mu, cov