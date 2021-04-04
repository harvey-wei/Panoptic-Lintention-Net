import torch
import torch.nn as nn
import unittest

class Conv1x1_unittest(unittest.TestCase):
    def test_conv1x1(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}")
        N, C, H, W = 4, 3, 124, 68
        in_feats = torch.rand(N, C, H, W, device=device)
        conv1x1 = nn.Conv2d(C, C//2, kernel_size=1).to(device=device)
        out = conv1x1(in_feats)
        self.assertEqual(out.size(), (N, C//2, H, W))
        loss = out.sum()
        loss.backward()


if __name__ == "__main__":
    unittest.main()