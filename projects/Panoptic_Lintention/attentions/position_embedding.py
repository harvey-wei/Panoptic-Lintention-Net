import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    This position embedding is adapted from that in the paper: The attention is all you need, generalized to
    to work on images by distribute half the d_model to height and width dimension, respectively.

    Note: Here we return the sum of input feature map and its positional embedding.
    """

    def __init__(self, d_model_half, temperature=10000):
        """
        Args:
            d_model_half: Half the depth of the feature map.
        """
        super(PositionEmbeddingSine, self).__init__()
        self.d_model_half = d_model_half
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        """x is of shape (N, H, W, C)"""
        n, h, w, c = x.size()
        assert c == 2 * self.d_model_half

        # pe_h shape (h, d_model_half)
        # pe_w shape (w, d_model_half)
        pe_h = torch.zeros(h, self.d_model_half)
        pe_w = torch.zeros(w, self.d_model_half)

        # position_h shape (h, 1)
        # position_w shape (w, 1)
        position_h = torch.arange(0, h, dtype=x.dtype, device=x.device).unsqueeze(1)
        position_w = torch.arange(0, w, dtype=x.dtype, device=x.device).unsqueeze(1)

        # div_term shape (d_model_half // 2)
        div_term = torch.exp(torch.arange(0, self.d_model_half, 2).float() * (-math.log(self.temperature)
                                                                              / self.d_model_half))
        div_term = div_term.to(device=x.device)
        # (h, d_model_half//2) is broadcastable with (d_model_half//2)
        pe_h[:, 0::2] = torch.sin(position_h * div_term)  # even model dim
        pe_h[:, 1::2] = torch.cos(position_h * div_term)  # odd model dim
        pe_w[:, 0::2] = torch.sin(position_w * div_term)  # even model dim
        pe_w[:, 1::2] = torch.cos(position_w * div_term)  # odd model dim

        # position_h shape ->(1, h, 1, 1)->(n, h, w, d_model_half)
        position_h = position_h.unsqueeze(0).unsqueeze(2).expand(n, h, w, self.d_model_half)

        # position_w shape ->(1, 1, w, 1)->(n, h, w, d_model_half)
        position_w = position_w.unsqueeze(0).unsqueeze(1).expand(n, h, w, self.d_model_half)

        # position shape (n, h, w, c)
        position = torch.cat((position_h, position_w), dim=-1)

        return x + position
