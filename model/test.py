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

def passing_argument(args):
    global down_dim
    global position_embedding_layer
    global do_local_self_attn
    global window_size

    down_dim = args.down_dim
    position_embedding_layer = args.position_embedding_layer
    window_size = args.window_size
    do_local_self_attn = args.do_local_self_attn


def add_attn_argument(parser: argparse.ArgumentParser) :
    parser.add_argument("--down_dim", type=int, default=160)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
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
def reshape_heads_to_batch_dim(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
    return tensor
def reshape_batch_dim_to_heads(tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = 8
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor

to_q = nn.Linear(320, 320)
to_k = nn.Linear(320, 320)
to_v = nn.Linear(320, 320)
window_size = 4

def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None):

    query = to_q(hidden_states)
    context = context if context is not None else hidden_states
    key = to_k(context)
    value = to_v(context)
    query = reshape_heads_to_batch_dim(query)
    key = reshape_heads_to_batch_dim(key)
    value = reshape_heads_to_batch_dim(value)


    local_hidden_states = localize_hidden_states(hidden_states, window_size)
    window_num = int(local_hidden_states.shape[0] / hidden_states.shape[0])
    local_query = to_q(local_hidden_states)
    local_key = to_k(local_hidden_states)
    local_value = to_v(local_hidden_states)

    local_query = reshape_heads_to_batch_dim(local_query)
    local_key = reshape_heads_to_batch_dim(local_key)
    local_value = reshape_heads_to_batch_dim(local_value)

    attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device), query,
                                     key.transpose(-1, -2), beta=0 )
    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(value.dtype)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = reshape_batch_dim_to_heads(hidden_states)
    #hidden_states = self.to_out[0](hidden_states)


    local_attention_scores = torch.baddbmm(torch.empty(local_query.shape[0], local_query.shape[1], local_key.shape[1],
                                 dtype=local_query.dtype, device=local_query.device), local_query,
                                 local_key.transpose(-1, -2), beta=0)
    local_attention_probs = local_attention_scores.softmax(dim=-1)
    local_attention_probs = local_attention_probs.to(local_value.dtype)
    local_hidden_states = torch.bmm(local_attention_probs, local_value)
    local_hidden_states = einops.rearrange(local_hidden_states, '(h w) p d -> h (p w) d', h=8, w=window_num)
    local_hidden_states = reshape_batch_dim_to_heads(local_hidden_states)

    hidden_states = hidden_states + local_hidden_states

    return hidden_states

hidden_states = torch.randn(1, 64*64, 320)
a = forward(hidden_states)
print(a.shape)