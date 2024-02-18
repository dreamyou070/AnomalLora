import torch.nn as nn
from typing import List
def save_tensors(module: nn.Module, features, name: str):
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, out):
    save_tensors(self, out, 'activations')
    return out


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module,
                 blocks: List[int],
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        # Save decoder activations
        for block_idx, block in enumerate(model.up_blocks):
            if block_idx != 0:
                attention_blocks = block.attentions
                resnet_blocks = block.resnets
                block_pair = [(res, attn) for res, attn in zip(resnet_blocks, attention_blocks)]
            else:
                block_pair = block.resnets
            for i, sub_block in enumerate(block_pair):
                if type(sub_block) == tuple:
                    sub_block = sub_block[-1]
                sub_block.register_forward_hook(save_out_hook)
                self.feature_blocks.append(block)
