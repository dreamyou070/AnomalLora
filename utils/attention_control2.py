from torch import nn
from data.perlin import rand_perlin_2d_np
import torch
from attention_store import AttentionStore
import argparse
import einops

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




def add_attn_argument(parser: argparse.ArgumentParser) :
    parser.add_argument("--down_dim", type=int, default=160)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def localize_hidden_states(hidden_states, window_size):
    b, p, d = hidden_states.shape
    res = int(p ** 0.5)
    hidden_states = hidden_states.view(b, res, res, d)
    local_hidden_states = window_partition(hidden_states, window_size).view(-1, window_size * window_size, d)
    return local_hidden_states

def passing_argument(args):
    global down_dim
    global position_embedding_layer
    global do_local_self_attn
    global only_local_self_attn
    global fixed_window_size
    global do_add_query
    global argument

    down_dim = args.down_dim
    position_embedding_layer = args.position_embedding_layer
    do_local_self_attn = args.do_local_self_attn
    only_local_self_attn = args.only_local_self_attn
    fixed_window_size = args.fixed_window_size
    do_add_query = args.do_add_query
    argument = args


def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None):

            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            if layer_name == position_embedding_layer :
                hidden_states_pos = noise_type(hidden_states)
                hidden_states = hidden_states_pos

            query = self.to_q(hidden_states)
            if trg_layer_list is not None and layer_name in trg_layer_list :
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

            if not is_cross_attention and do_local_self_attn :

                if not fixed_window_size :
                    H = int(hidden_states.shape[1] ** 0.5)
                    window_size = int(H / 2)
                else :
                    window_size = argument.window_size

                local_hidden_states = localize_hidden_states(hidden_states, window_size)
                window_num = int(local_hidden_states.shape[0] / hidden_states.shape[0])

                local_query = self.to_q(local_hidden_states)
                local_key = self.to_k(local_hidden_states)
                local_value = self.to_v(local_hidden_states)

                local_query = self.reshape_heads_to_batch_dim(local_query)
                local_key = self.reshape_heads_to_batch_dim(local_key)
                local_value = self.reshape_heads_to_batch_dim(local_value)

                if self.upcast_attention:
                    local_query = local_query.float()
                    local_key = local_key.float()
            if do_add_query:

                #if layer_name == position_embedding_layer :
                #    controller.save_query_sub(query, layer_name)

                if layer_name in argument.add_query_layer_list :
                    controller.save_query_sub(query, layer_name)

            if do_add_query :
                if layer_name in trg_layer_list :
                    query_dict_sub = controller.query_dict_sub
                    for k in query_dict_sub.keys():
                        before_query = query_dict_sub[k][0]
                        query += before_query
                    controller.query_dict_sub = {}

            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device), query,
                                             key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)
            local_hidden_states_out = torch.zeros_like(hidden_states)

            if not is_cross_attention and do_local_self_attn :
                for window_index in range(window_num):
                    l_query = local_query[window_index * self.heads : (window_index + 1) * self.heads, :, :]
                    l_key = local_key[window_index * self.heads: (window_index + 1) * self.heads, :, :]
                    l_value = local_value[window_index * self.heads: (window_index + 1) * self.heads, :, :]
                    if l_query.dim() == 2:
                        l_query = l_query.unsqueeze(0)
                        l_key = l_key.unsqueeze(0)
                        l_value = l_value.unsqueeze(0)
                    l_attention_scores = torch.baddbmm(torch.empty(l_query.shape[0], l_query.shape[1], l_key.shape[1], dtype=l_query.dtype, device=l_query.device),
                                                       l_query, l_key.transpose(-1, -2),
                                                       beta=0, alpha=self.scale, ).softmax(dim=-1).to(l_value.dtype)
                    l_hidden_states = self.reshape_batch_dim_to_heads(torch.bmm(l_attention_scores, l_value))
                    l_hidden_states = self.to_out[0](l_hidden_states)
                    local_pix_num = l_hidden_states.shape[1]
                    local_hidden_states_out[:, window_index * local_pix_num: (window_index + 1) * local_pix_num, :] = l_hidden_states

            if trg_layer_list is not None and layer_name in trg_layer_list :
                trg_map = attention_probs[:, :, :2]
                controller.store(trg_map, layer_name)

            if not is_cross_attention and do_local_self_attn :
                if only_local_self_attn :
                    hidden_states = local_hidden_states_out
                else :
                    hidden_states = hidden_states + local_hidden_states_out
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