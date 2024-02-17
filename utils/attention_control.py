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

def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):

        def local_self_attn(hidden_states):

            B, L, C = hidden_states.shape
            H = W = int(L ** 0.5)
            hidden_states = hidden_states.view(B, H, W, C)
            # [2]  window partitioning
            x_windows = window_partition(hidden_states, argument.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C
            B_, N, C = x_windows.shape  # 4, 64, 320
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            query = query.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)  # batch_num / head / len / dim
            key = key.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
            value = value.reshape(B_, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
            if self.upcast_attention:
                query = query.float()
                key = key.float()
            attention_scores = (query @ key.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().to(attention_scores.device)
            attention_scores = attention_scores + relative_position_bias.unsqueeze(0)  # attn = [64, 3, 49, 49] + [1, 3, 49, 49]
            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            hidden_states = (attention_probs @ value).transpose(1, 2).reshape(B_, N,
                                                                              C)  # nW*B, window_size*window_size, C
            attn_windows = hidden_states.view(-1, self.window_size[0], self.window_size[1], C)  # 64, 7, 7, 96
            window_attn = window_reverse(attn_windows, self.window_size[0], H, W)  # batch, w_size, w_size, c
            hidden_states = window_attn.view(B, H * W, C)
            local_hidden_states = self.to_out[0](hidden_states)
            return local_hidden_states

        def global_self_attn(hidden_states, context, trg_layer_list, do_attn, object_attention_mask):

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
            #batch, pix_num, sen_len
            if do_attn :
                sen_len = attention_scores.shape[2]
                object_attention_mask = object_attention_mask.unsqueeze(-1).expand(1,1, sen_len)
                attention_scores = attention_scores + object_attention_mask

            attention_probs = attention_scores.softmax(dim=-1).to(value.dtype)
            hidden_states = torch.bmm(input=attention_probs, mat2=value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            global_hidden_states = self.to_out[0](hidden_states)
            return global_hidden_states, attention_scores, attention_probs


        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None, **model_kwargs):

            do_attn = False
            object_attention_mask = None
            print(f'model_kwargs: {model_kwargs}')
            print(f'type of model_kwargs: {type(model_kwargs)}')
            if 'object_attention_mask' in model_kwargs:
                object_attention_mask = model_kwargs['object_attntion_mask']
                b_, pix_num = object_attention_mask.shape
                hidden_states_pix_num = hidden_states.shape[1]
                if pix_num == hidden_states_pix_num:
                    do_attn = True


            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            # [1] position embedding
            if layer_name == argument.position_embedding_layer :
                hidden_states_pos = noise_type(hidden_states)
                hidden_states = hidden_states_pos

            if not is_cross_attention and argument.do_local_self_attn:
                local_hidden_states = local_self_attn(hidden_states,)
            global_hidden_states, attention_scores, attention_probs = global_self_attn(hidden_states,
                                                                                       context,
                                                                                       trg_layer_list,
                                                                                       do_attn,
                                                                                       object_attention_mask)

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

    def register_window_function(self,)  :

        def add_window_argument(window_size) :
            self.window_size = (window_size, window_size)
            num_heads = self.heads
            #head_dim = dim // num_heads
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))

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

        return add_window_argument(argument.window_size)

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, layer_name)
            register_window_function(net_)

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