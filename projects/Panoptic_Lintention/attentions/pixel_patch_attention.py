import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_embedding import PositionEmbeddingSine
import math


class PPAttention(nn.Module):
    """
    Pixel to patch attention module with residual connection
    Queries(N, H, W, C) is the projection of input_feats (N, H, W, C)
    Keys (N, P, C)
    Values (N, P, C)
    Outputs (N, H, W, C)
    """
    def __init__(self,
                 in_channels,
                 num_patches=16,
                 query_project=False,
                 patches_project=False,
                 position_embed=False,
                 ):
        super().__init__()
        # Let C = in_channels,
        # Let P = num_patches
        # Input shape: (N, H, W, C)
        # Output shape: (N, H, W, P)

        self.query_project = query_project
        self.patches_project = patches_project
        self.position_embed = position_embed

        self.classifier = nn.Sequential(nn.Linear(in_channels, num_patches, bias=False),
                                        nn.Softmax(dim=3))

        if position_embed:
            self.position_embedding = PositionEmbeddingSine(d_model_half=in_channels // 2)

        if query_project:
            # project input of shape (N, H, W, C) to the same shape
            self.query_mat = nn.Linear(in_channels, in_channels, bias=False)

        if patches_project:
            self.values_mat = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        """

        Args:
            x: input feature map of shape (N, H, W, C)

        Returns:
            tensor of shape (N, H, )

        """
        if self.position_embed:
            x = self.position_embedding(x)

        # x is of shape (N, H, W, C)
        C = x.shape[-1]

        # x:(N, W, W, C) -> queries:(N, H, W, C)
        queries = self.query_mat(x) if self.query_project else x

        # x:(N, H, W, C) -> y:(N, H, W, P)
        y = self.classifier(x)
        # queries:(N, H, W, C), y:(N, H, W, P)->
        # sem_patches: (N, P, C), i.e. keys
        keys = torch.einsum('nhwc,nhwp->npc', queries, y)

        # project keys into values
        # keys (N, P, C) -> values (N, P, C)
        values = self.values_mat(keys) if self.patches_project else keys

        # keys: (N, P, C), queries(N, H, W, C)
        # -> sim_mat: (N, H, W, P)
        sim_mat = torch.einsum('npc, nhwc->nhwp', keys, queries)
        # sim_mat /= torch.sqrt(C)  # normalized by length
        sim_mat = sim_mat / (C ** 0.5)
        sim_mat = F.softmax(sim_mat, dim=-1)  #

        # import matplotlib.pyplot as plt
        # import numpy as np
        # self.sim_mat = sim_mat.clone() # for hook.
        # # self.sim_mat = self.sim_mat.sum(dim=3)
        # att_map = self.sim_mat.cpu().detach().numpy()
        # plt.imshow(att_map[0, :, :, 14], cmap='gray')
        # plt.show()

        # outputs = torch.einsum('nhwp, npc->nhwc', sim_mat, values)
        # outputs = outputs.sum(dim=3)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # outs = outputs.cpu().detach().numpy()
        # plt.imshow(outs[0, :, :])
        # plt.show()

        # Let sem_patches be the values
        # sim_mat: (N, H, W, P), values: (N, P, C) -> (N, H, W, C)
        return torch.einsum('nhwp, npc->nhwc', sim_mat, values)


class MultiHeadPPAttention(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads=8,
                 num_patches=16,
                 query_project=False,
                 patches_project=False,
                 position_embed=False
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (self.head_dim * num_heads == d_model), "Channel size needs to be divisible by number of heads."

        self.heads = nn.ModuleList(
            nn.ModuleDict({
                'dim_reduction' + str(i): nn.Linear(self.d_model, self.head_dim, bias=False),
                'PPAttention' + str(i): PPAttention(in_channels=self.head_dim,
                                                    num_patches=num_patches,
                                                    query_project=query_project,
                                                    patches_project=patches_project,
                                                    position_embed=position_embed
                                                    )
            })
            for i in range(self.num_heads)
        )

        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: input feature map of shape (N, H, W, C) with C = d_model

        Returns:
            tensor of shape (N, H, W, C)
        """
        heads = []
        for i in range(self.num_heads):
            # input shape (N, H, W, d_model)
            # output shape (N, H, W, head_dim)
            head_input = self.heads[i]['dim_reduction' + str(i)](x)

            # output shape (N, H, W, head_dim)
            heads.append(self.heads[i]['PPAttention' + str(i)](head_input))

        # multi_head shape (N, H, W, d_model)
        multi_head = torch.cat(heads, dim=-1)

        return self.fc_out(multi_head)


class PPALayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 num_patches,
                 query_project=False,
                 patches_project=False,
                 position_embed=False,
                 dropout=0.1,
                 ffn_expansion=4,
                 ):
        super().__init__()

        self.attention = MultiHeadPPAttention(d_model=d_model,
                                              num_heads=num_heads,
                                              num_patches=num_patches,
                                              query_project=query_project,
                                              patches_project=patches_project,
                                              position_embed=position_embed,
                                              )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * ffn_expansion),
                                 nn.ReLU(),
                                 nn.Linear(d_model * ffn_expansion, d_model)
                                 )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: input feature map of shape (N, H, W, C) with C as d_model

        Returns:
            a tensor of shape (N, H, W, C)
        """
        # att shape (N, H, W, C)
        att = self.attention(x)

        # Residual connection followed by layer norm and dropout
        # out_heads shape (N, H, W, C)
        out_heads = self.dropout(self.layer_norm1(x + att))

        # out_ffn shape (N, H, W, C)
        out_ffn = self.ffn(out_heads)

        out = self.dropout(self.layer_norm2(out_heads + out_ffn))

        return out

