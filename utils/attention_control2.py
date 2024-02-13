from torch import nn
from data.perlin import rand_perlin_2d_np
import torch
from attention_store import AttentionStore
import argparse

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
    global do_concat
    global do_double_selfattn

    down_dim = args.down_dim
    position_embedding_layer = args.position_embedding_layer
    do_concat = args.do_concat
    do_double_selfattn = args.do_double_selfattn

def add_attn_argument(parser: argparse.ArgumentParser) :
    parser.add_argument("--down_dim", type=int, default=160)


def register_attention_control(unet: nn.Module,controller: AttentionStore):

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, trg_layer_list=None, noise_type=None):

            is_cross_attention = False
            if context is not None:
                is_cross_attention = True

            if layer_name == position_embedding_layer:  # 'down_blocks_0_attentions_0_transformer_blocks_0_attn1' : """ Position Embedding right after Down Block 1 """
                hidden_states_pos = noise_type(hidden_states)
                hidden_states = hidden_states_pos

            query = self.to_q(hidden_states)

            if trg_layer_list is not None and layer_name in trg_layer_list :
                controller.save_query(query, layer_name)

            if do_double_selfattn :
                if is_cross_attention:
                    self_key = self.to_k(hidden_states)
                    self_value = self.to_v(hidden_states)
                    self_query = self.reshape_heads_to_batch_dim(query)
                    self_key = self.reshape_heads_to_batch_dim(self_key)
                    self_value = self.reshape_heads_to_batch_dim(self_value)
                    self_attention_scores = torch.baddbmm(torch.empty(self_query.shape[0], self_query.shape[1], self_key.shape[1],dtype=query.dtype, device=query.device),
                                                          self_query, self_key.transpose(-1, -2),
                                                          beta=0, alpha=self.scale, )
                    self_attention_probs = self_attention_scores.softmax(dim=-1).to(query.dtype)
                    self_hidden_states = torch.bmm(self_attention_probs, self_value)
                    self_hidden_states = self.reshape_batch_dim_to_heads(self_hidden_states)
                    hidden_states = self.reshape_batch_dim_to_heads(self_hidden_states)

            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            query = self.reshape_heads_to_batch_dim(query)
            # if layer_name == position_embedding_layer and do_concat :
            #    query_pos = self.reshape_heads_to_batch_dim(query_pos)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            if self.upcast_attention:
                query = query.float()
                key = key.float()
                self_key = self_key.float()

            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1],
                                                         dtype=query.dtype, device=query.device), query,
                                             key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)

            if trg_layer_list is not None and layer_name in trg_layer_list :
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