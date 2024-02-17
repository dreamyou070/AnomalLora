from torch import nn
from data.perlin import rand_perlin_2d_np
import torch
from attention_store import AttentionStore
import argparse
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def passing_argument(args):
    global down_dim
    global do_local_self_attn
    global only_local_self_attn
    global fixed_window_size
    global argument

    down_dim = args.down_dim
    argument = args

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)                   # (320, 320)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)         # (768, 320)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)         # (768, 320)
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        # no dropout here

        self.use_memory_efficient_attention_xformers = False
        self.use_memory_efficient_attention_mem_eff = False
        self.use_sdpa = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        self.use_memory_efficient_attention_xformers = xformers
        self.use_memory_efficient_attention_mem_eff = mem_eff

    def set_use_sdpa(self, sdpa):
        self.use_sdpa = sdpa

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, trg_layer_list=None, noise_type=None):
        if self.use_memory_efficient_attention_xformers:
            return self.forward_memory_efficient_xformers(hidden_states, context, mask)
        if self.use_memory_efficient_attention_mem_eff:
            return self.forward_memory_efficient_mem_eff(hidden_states, context, mask)
        if self.use_sdpa:
            return self.forward_sdpa(hidden_states, context, mask)

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

    # TODO support Hypernetworks
    def forward_memory_efficient_xformers(self, x, context=None, mask=None):
        import xformers.ops

        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward_memory_efficient_mem_eff(self, x, context=None, mask=None):
        flash_func = FlashAttentionFunction
        q_bucket_size = 512
        k_bucket_size = 1024
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k = self.to_k(context)
        v = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out[0](out)
        return out

    def forward_sdpa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out[0](out)
        return out


def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):

        def add_window_argument(window_size) :
            self.window_size = window_size
            num_heads = self.heads
            #head_dim = dim // num_heads
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)


        def local_self_attn(hidden_states):

            B, L, C = hidden_states.shape
            H = W = int(L ** 0.5)
            hidden_states = hidden_states.view(B, H, W, C)
            # [2]  window partitioning
            x_windows = window_partition(hidden_states, argument.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, argument.window_size * argument.window_size,
                                       C)  # nW*B, window_size*window_size, C
            B_, N, C = x_windows.shape  # 4, 64, 320
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            query = query.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1,
                                                                              3)  # batch_num / head / len / dim
            key = key.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
            value = value.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = (query @ key.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attention_scores = attention_scores + relative_position_bias.unsqueeze(0)  # attn = [64, 3, 49, 49] + [1, 3, 49, 49]
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            hidden_states = (attention_probs @ value).transpose(1, 2).reshape(B_, N,
                                                                              C)  # nW*B, window_size*window_size, C
            attn_windows = hidden_states.view(-1, argument.window_size, argument.window_size, C)  # 64, 7, 7, 96
            window_attn = window_reverse(attn_windows, argument.window_size, H, W)  # batch, w_size, w_size, c
            hidden_states = window_attn.view(B, H * W, C)
            local_hidden_states = self.to_out[0](hidden_states)
            return local_hidden_states

        def global_self_attn(hidden_states, context, trg_layer_list):

            query = self.to_q(hidden_states)

            if trg_layer_list is not None and layer_name in trg_layer_list:
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
            hidden_states = torch.bmm(input=attention_probs, mat2=value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            global_hidden_states = self.to_out[0](hidden_states)
            return global_hidden_states, attention_scores, attention_probs


        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            # [1] position embedding
            if layer_name == argument.position_embedding_layer :
                hidden_states_pos = noise_type(hidden_states)
                hidden_states = hidden_states_pos

            if not is_cross_attention and argument.do_local_self_attn:
                local_hidden_states = local_self_attn(hidden_states)
            global_hidden_states, attention_scores, attention_probs = global_self_attn(hidden_states,
                                                                                       context,
                                                                                       trg_layer_list)

            if is_cross_attention :
                hidden_states = global_hidden_states
            else :
                if argument.do_local_self_attn :
                    if argument.only_local_self_attn :
                        hidden_states = local_hidden_states
                    else :
                        hidden_states = local_hidden_states + global_hidden_states
                else :
                    hidden_states = global_hidden_states


            if trg_layer_list is not None and layer_name in trg_layer_list :
                if argument.use_focal_loss :
                    attention_scores = attention_scores[:, :, :2]
                    attention_probs = attention_scores.softmax(dim=-1)
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
            net_.add_window_argument(argument.window_size)
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