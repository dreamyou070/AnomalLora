import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def mahal(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov, delta))
    return torch.sqrt(m)

def gen_mahal_loss(args, anormal_feat_list, normal_feat_list):

    normal_feats = torch.cat(normal_feat_list, dim=0)

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
        normal_dist_loss = normal_dist_max / total_dist

    else:
        normal_dist_loss = normal_dist_max
    normal_dist_loss = normal_dist_loss * args.dist_loss_weight

    return normal_dist_max, normal_dist_loss

def generate_attention_loss(attn_score, normal_position, do_calculate_anomal):

    device = attn_score.device

    normal_position = normal_position.to(device)

    # [1] preprocessing
    cls_score, trigger_score = attn_score.chunk(2, dim=-1)
    cls_score, trigger_score = cls_score.squeeze(), trigger_score.squeeze()  # head, pix_num
    cls_score, trigger_score = cls_score.mean(dim=0), trigger_score.mean(dim=0)  # pix_num
    # [2]
    normal_cls_score, normal_trigger_score = cls_score * normal_position, trigger_score * normal_position
    anomal_cls_score, anomal_trigger_score = cls_score * (1 - normal_position), trigger_score * (1 - normal_position)
    total_score = torch.ones_like(cls_score).to(device)
    normal_cls_score, normal_trigger_score = normal_cls_score / total_score, normal_trigger_score / total_score
    anomal_cls_score, anomal_trigger_score = anomal_cls_score / total_score, anomal_trigger_score / total_score
    # [3] loss calculating
    normal_trigger_loss = (1 - normal_trigger_score) ** 2
    normal_cls_loss = normal_cls_score ** 2

    anomal_trigger_loss, anomal_cls_loss = 0, 0
    if do_calculate_anomal:
        anomal_trigger_loss = (1 - anomal_trigger_score) ** 2
        anomal_cls_loss = anomal_cls_score ** 2

    return normal_trigger_loss.to(device), normal_cls_loss.to(device), anomal_trigger_loss.to(device), anomal_cls_loss.to(device)

def gen_value_dict(value_dict,
                   normal_trigger_loss, normal_cls_loss,
                   anormal_trigger_loss, anormal_cls_loss, ):
    if normal_cls_loss is not None:
        if 'normal_cls_loss' not in value_dict.keys():
            value_dict['normal_cls_loss'] = []
        value_dict['normal_cls_loss'].append(normal_cls_loss)
    if anormal_cls_loss is not None:
        if 'anormal_cls_loss' not in value_dict.keys():
            value_dict['anormal_cls_loss'] = []
        value_dict['anormal_cls_loss'].append(anormal_cls_loss)
    if normal_trigger_loss is not None:
        if 'normal_trigger_loss' not in value_dict.keys():
            value_dict['normal_trigger_loss'] = []
        value_dict['normal_trigger_loss'].append(normal_trigger_loss)
    if anormal_trigger_loss is not None:
        if 'anormal_trigger_loss' not in value_dict.keys():
            value_dict['anormal_trigger_loss'] = []
        value_dict['anormal_trigger_loss'].append(anormal_trigger_loss)
    return value_dict


def gen_attn_loss(value_dict):
    normal_cls_loss = torch.stack(value_dict['normal_cls_loss'], dim=0).mean(dim=0)
    anormal_cls_loss = torch.stack(value_dict['anormal_cls_loss'], dim=0).mean(dim=0)
    normal_trigger_loss = torch.stack(value_dict['normal_trigger_loss'], dim=0).mean(dim=0)
    anormal_trigger_loss = torch.stack(value_dict['anormal_trigger_loss'], dim=0).mean(dim=0)
    return normal_cls_loss, normal_trigger_loss, anormal_cls_loss, anormal_trigger_loss

def generate_anomal_map_loss(args, attn_score, normal_position, loss_focal, loss_l2):

    device = attn_score.device

    trigger_score = attn_score[:, :, 1].squeeze(0) # 8, pix_num, 1
    head_num = trigger_score.shape[0]
    res = int(trigger_score.shape[1] ** 0.5)
    trigger_score = trigger_score.view(head_num, res, res).unsqueeze(1)

    cls_score = attn_score[:, :, 0].squeeze(0)
    cls_score = cls_score.view(head_num, res, res).unsqueeze(1)

    normal_position = normal_position.view(1, 1, res, res)
    normal_position = normal_position.repeat(head_num, 1, 1, 1).to(device) # 8, 1, 64, 64

    if args.use_focal_loss:
        focal_loss_in = torch.cat([cls_score, trigger_score], 1) # 8, 2, 64,64
        focal_loss_trg = 1 - normal_position
        if args.adv_focal_loss:
            focal_loss_trg = 1 - focal_loss_trg
        loss = loss_focal(focal_loss_in, focal_loss_trg)
    else:
        loss = loss_l2(trigger_score.float(),
                       normal_position.float())
    print(f'in anomal map loss, loss: {loss}')
    print(f'type of loss: {type(loss)}')

    return loss

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):

    if val_range is None:

        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val # decide value range
    else:
        l = val_range

    padd = window_size//2

    (_, channel, height, width) = img1.size()

    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

    def forward(self, img1, img2):

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window    # if channel == 1, window  = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2,
                                 window=window,
                                 window_size=self.window_size,
                                 size_average=self.size_average)
        return 1.0 - s_score