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
    global do_local_self_attn
    global only_local_self_attn
    global fixed_window_size
    global argument

    down_dim = args.down_dim
    argument = args


def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            if layer_name == argument.position_embedding_layer :
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

            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1],
                                                         dtype=query.dtype, device=query.device), query,
                                             key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            hidden_states = self.to_out[0](hidden_states)

            if trg_layer_list is not None and layer_name in trg_layer_list :
                if argument.use_focal_loss :
                    attention_scores = attention_scores[:, :, :2]
                    attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
                    trg_map = attention_probs[:, :, :2]
                    controller.store(trg_map, layer_name)
                else :
                    trg_map = attention_probs[:, :, :2]
                    controller.store(trg_map, layer_name)

            if layer_name == argument.image_classification_layer :
                controller.store_classifocation_map(attention_probs[:, :, 1], layer_name)
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