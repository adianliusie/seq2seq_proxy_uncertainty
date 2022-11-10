from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import pytorch_wrapper as pw

from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


__all__ = [
    "load_model"
]


def load_model(system: str) -> T5ForConditionalGeneration:
    """
    Downloads and returns the relevant pretrained seq2seq transformer from huggingface
    """
    if system == 't5-small':
        return T5ForConditionalGeneration.from_pretrained("t5-small", return_dict = True)
    elif system == 't5-base': 
        return T5ForConditionalGeneration.from_pretrained("t5-base", return_dict = True)
    elif system == 't5-large': 
        return T5ForConditionalGeneration.from_pretrained("t5-large", return_dict = True)
    elif system == 't5-xl': 
        return T5ForConditionalGeneration.from_pretrained("t5-3b", return_dict = True)
    if system == 't5-small-proxy':
        return T5ProxyForConditionalGeneration.from_pretrained("t5-small", return_dict = True)
    elif system == 't5-base-proxy': 
        return T5ProxyForConditionalGeneration.from_pretrained("t5-base", return_dict = True)
    elif system == 't5-large-proxy': 
        return T5ProxyForConditionalGeneration.from_pretrained("t5-large", return_dict = True)
    if system == 't5-small-proxy-rev':
        return T5ProxyRevForConditionalGeneration.from_pretrained("t5-small", return_dict = True)
    elif system == 't5-base-proxy-rev': 
        return T5ProxyRevForConditionalGeneration.from_pretrained("t5-base", return_dict = True)
    elif system == 't5-large-proxy-rev': 
        return T5ProxyRevForConditionalGeneration.from_pretrained("t5-large", return_dict = True)

    raise ValueError("{} is an invalid system: no seq2seq model found".format(system))



class ProxyHead(nn.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, reverse: bool = False):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, intermediate_dim)
        self.lin2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.lin3 = nn.Linear(intermediate_dim, 1)
        self.reverse = reverse

    def invariance(self, x: torch.Tensor) -> torch.Tensor:
        return x.sort(dim = -1).values

    def forward(self, x: torch.Tensor, mask: torch.Tensor, sorting: bool = False) -> torch.Tensor:
        
        # When reverse is false we perform pooling prior to feeding into the head
        x = x if self.reverse else pw.functional.masked_mean_pooling(x, mask, dim = 1)
        
        # The smaller mlp
        x = torch.softmax(self.lin1(x), dim = -1)
        x = self.invariance(x) if sorting else x
        x = torch.tanh(self.lin2(x))
        x = self.lin3(x).squeeze(-1)

        # When reverse is activated 
        x = x if not self.reverse else pw.functional.masked_mean_pooling(x, mask, dim = 1)
        return x


class T5ProxyForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

        # Build encoder head
        self.encoder_head = ProxyHead(
            input_dim = config.d_model,
            intermediate_dim = config.d_ff,
        )

    def set_arguments(self, args):
        self.uargs = args

    # encoder_hidden_states
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        # Forward propagate through main model
        output = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            **kwargs,
        )

        # Get the proxy scores as well
        output.proxies = self.encoder_head(
            output.encoder_last_hidden_state, 
            mask = attention_mask,
            sorting = self.uargs.proxy_permutation_invariance,
        )

        return output


class T5ProxyRevForConditionalGeneration(T5ProxyForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

        # Perform pooling post encoder-head
        self.encoder_head.reverse = True