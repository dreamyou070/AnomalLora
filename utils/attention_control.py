from torch import nn
from data.perlin import rand_perlin_2d_np
import torch
from attention_store import AttentionStore

def register_attention_control(unet: nn.Module,
                               controller: AttentionStore, ):  # if mask_threshold is 1, use itself

    def ca_forward(self, layer_name):

        def forward(hidden_states, context=None, trg_indexs_list=None, mask=None):

            query = self.to_q(hidden_states)
            if trg_indexs_list is not None and layer_name in trg_indexs_list:
                b = hidden_states.shape[0]
                if b == 1 :
                    pix_num, dim = hidden_states.shape[1], hidden_states.shape[2]
                    if mask : # mask means using perlin noise
                        perlin_scale = 6
                        min_perlin_scale = 0
                        rand_1 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
                        rand_2 = torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0]
                        perlin_scalex, perlin_scaley = 2 ** (rand_1), 2 ** (rand_2)
                        perlin_noise = rand_perlin_2d_np((pix_num, dim), (perlin_scalex, perlin_scaley))
                        perlin_noise = torch.tensor(perlin_noise).to(hidden_states.device)
                        anomal_query = self.to_q(perlin_noise)
                        temp_query = torch.cat([query, anomal_query], dim=0)
                        controller.save_query(temp_query, layer_name)
                        rand_query = self.reshape_heads_to_batch_dim(temp_query)
                        if self.upcast_attention:
                            temp_query = temp_query.float()
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
            attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            # cast back to the original dtype
            attention_probs = attention_probs.to(value.dtype)
            if trg_indexs_list is not None and layer_name in trg_indexs_list:
                if b == 1 :
                    temp_key = torch.cat([key, key], dim=0)
                    temp_attention_scores = torch.baddbmm(torch.empty(temp_query.shape[0], temp_query.shape[1],
                                                                 temp_key.shape[1], dtype=query.dtype, device=query.device),
                                                     temp_query, temp_key.transpose(-1, -2), beta=0,alpha=self.scale, )
                    temp_attention_probs = temp_attention_scores.softmax(dim=-1)
                    temp_trg_map = temp_attention_probs.to(value.dtype)[:, :, :2]
                    controller.store(temp_trg_map, layer_name)
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
