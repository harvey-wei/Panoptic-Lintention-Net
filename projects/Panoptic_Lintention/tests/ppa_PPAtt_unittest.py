from projects.Panoptic_Attention.attentions import (
    PPAttention,
    PositionEmbeddingSine,
    MultiHeadPPAttention,
    PPALayer,
)
import unittest
import torch


class PPAttentionTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N, C, H, W = 8, 128, 512 // 32, 1024 // 32
        self.input_feats = torch.rand((N, C, H, W), device=self.device)

    def test_PosEmbedSine(self):
        N, C, H, W = self.input_feats.size()
        self.input_feats = self.input_feats.permute(0, 2, 3, 1)
        pe = PositionEmbeddingSine(d_model_half=C // 2)
        pe = pe.to(device=self.device)
        res = pe(self.input_feats)
        self.assertEqual(res.shape, (N, H, W, C))

    def test_PPAtt(self):
        N, C, H, W = self.input_feats.size()
        self.input_feats = self.input_feats.permute(0, 2, 3, 1)
        # ppa = PPAttention(C, 8)
        ppa = PPAttention(C, 8, query_project=True, patches_project=True, position_embed=True)
        ppa = ppa.to(device=self.device)
        out = ppa(self.input_feats)
        self.assertEqual(out.shape, (N, H, W, C))
        loss = out.sum()
        loss.backward()
        print(f'device {self.device}')

    def test_MultiHeadPPAtt(self):
        N, C, H, W = self.input_feats.size()
        self.input_feats = self.input_feats.permute(0, 2, 3, 1)
        num_heads = 8
        multi_head_ppa = MultiHeadPPAttention(d_model=C, num_heads=num_heads, query_project=True,
                                              patches_project=True, position_embed=True)
        multi_head_ppa = multi_head_ppa.to(device=self.device)
        out = multi_head_ppa(self.input_feats)
        self.assertEqual(out.shape, (N, H, W, C))
        loss = out.sum()
        loss.backward()

    def test_PPAEncoderLayer(self):
        N, C, H, W = self.input_feats.size()
        self.input_feats = self.input_feats.permute(0, 2, 3, 1)
        num_heads = 8
        ppa_coder_layer = PPALayer(d_model=C,
                                   num_heads=num_heads,
                                   num_patches=8,
                                   query_project=True,
                                   patches_project=True,
                                   position_embed=True
                                   )
        ppa_coder_layer.to(device=self.device)
        out = ppa_coder_layer(self.input_feats)
        self.assertEqual(out.shape, (N, H, W, C))
        loss = out.sum()
        loss.backward()


if __name__ == '__main__':
    unittest.main()
