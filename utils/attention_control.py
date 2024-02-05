from torch import nn
from data.perlin import rand_perlin_2d_np
import torch
from attention_store import AttentionStore

def mahal(u, v, cov):
    delta = u - v
    cov_inv = cov.T
    m_ = torch.matmul(cov_inv, delta)
    m = torch.dot(delta, m_)
    return torch.sqrt(m)

def make_perlin_noise(shape_row, shape_column):
    perlin_scale = 6
    min_perlin_scale = 0
    rand_1 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    rand_2 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
    perlin_scalex, perlin_scaley = 2 ** (rand_1), 2 ** (rand_2)
    perlin_noise = rand_perlin_2d_np((shape_row, shape_column), (perlin_scalex, perlin_scaley))
    return perlin_noise



def register_attention_control(unet: nn.Module,controller: AttentionStore, ):  # if mask_threshold is 1, use itself

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):

            query = self.to_q(hidden_states)
            if trg_indexs_list is not None and layer_name in trg_indexs_list:
                b = hidden_states.shape[0]
                if b == 1 :
                    normal_query = query.squeeze(0)

                    pix_num, dim = normal_query.shape[0], normal_query.shape[1]
                    normal_feats = []
                    for pix_idx in range(pix_num):
                        normal_feat = normal_query[pix_idx].squeeze(0)
                        normal_feats.append(normal_feat.unsqueeze(0))
                    normal_feats = torch.cat(normal_feats, dim=0)

                    """ random down sampling the dim """
                    from random import sample
                    down_dim = 100
                    idx = torch.tensor(sample(range(0, dim), down_dim)).to(hidden_states.device)
                    # print(idx)
                    normal_feats = torch.index_select(normal_feats, 1, idx)
                    normal_mu = torch.mean(normal_feats, dim=0)
                    normal_cov = torch.cov(normal_feats.transpose(0, 1))



                    # ---------------------------------------------------------------------------------------------- #
                    normal_mahalanobis_dists = [mahal(feat, normal_mu, normal_cov) for feat in normal_feats]
                    max_dist = max(normal_mahalanobis_dists)
                    #mean_dist = torch.mean(torch.tensor(normal_mahalanobis_dists))
                    th = max_dist.item() * 0.8
                    # ---------------------------------------------------------------------------------------------- #
                    if mask == 'perlin' : # mask means using perlin noise
                        perlin_noise = make_perlin_noise(pix_num, dim)
                        perlin_noise = torch.tensor(perlin_noise).to(hidden_states.device)
                        noise = hidden_states.squeeze() + perlin_noise
                    else :
                        noise = torch.randn_like(hidden_states).to(hidden_states.device)
                    anomal_map, anomal_features = [], []
                    normal_hidden_states = hidden_states.squeeze(0)
                    anormal_hidden_states = noise.squeeze(0)
                    for pix_idx in range(pix_num):
                        sub_feature = anormal_hidden_states[pix_idx, :].squeeze(0)
                        down_dim_sub_feature = torch.index_select(sub_feature.unsqueeze(0), 1, idx)
                        normal_feat = normal_hidden_states[pix_idx, :].squeeze(0)
                        sub_dist = mahal(down_dim_sub_feature.float().squeeze(), normal_mu.squeeze(), normal_cov)
                        if sub_dist > th.item():
                            anomal_features.append(sub_feature.unsqueeze(0))
                            anomal_map.append(1)
                        else:
                            # ----------------------------------------------------------------------------------------- #
                            # append normal feature
                            anomal_features.append(normal_feat.unsqueeze(0))
                            anomal_map.append(0)
                    anormal_hidden_states = torch.cat(anomal_features, dim=0).to(hidden_states.dtype) # pix_num, dim
                    anomal_map = torch.tensor(anomal_map).unsqueeze(0)
                    res = int(pix_num ** 0.5)
                    anomal_map = anomal_map.view(res, res)
                    # ---------------------------------------------------------------------------------------------- #
                    temp_query = torch.cat([query, self.to_q(anormal_hidden_states.unsqueeze(0))], dim=0)
                    controller.save_query(temp_query, layer_name) # [2, res*res, 320]
                    controller.save_query([normal_mu, normal_cov], layer_name)
                    controller.save_map(anomal_map, layer_name)   # [res,res]
                    temp_query = self.reshape_heads_to_batch_dim(temp_query)
                    if self.upcast_attention:
                        temp_query = temp_query.float()
                    # ---------------------------------------------------------------------------------------------- #
                else :
                    controller.save_query(query, layer_name)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype,
                                                         device=query.device), query,key.transpose(-1, -2),
                                             beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if trg_indexs_list is not None and layer_name in trg_indexs_list:
                if b == 1 :
                    temp_key = torch.cat([key, key], dim=0)
                    temp_attention_scores = torch.baddbmm(torch.empty(temp_query.shape[0], temp_query.shape[1],
                                                                 temp_key.shape[1], dtype=query.dtype, device=query.device),
                                                     temp_query, temp_key.transpose(-1, -2), beta=0,alpha=self.scale, )
                    temp_attention_probs = temp_attention_scores.softmax(dim=-1)
                    temp_trg_map = temp_attention_probs.to(value.dtype)[:, :, :2]
                    controller.store(temp_trg_map, layer_name) # 2, res*res, 2
                else :
                    trg_map = attention_probs[:, :, :2]
                    controller.store(trg_map, layer_name)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states
        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count


    def forward(self, hidden_states, context=None, trg_indexs_list=None, mask=None):

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)     # no dropout
        return hidden_states

    def _attention(self, query, key, value):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,)
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
