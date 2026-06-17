"""
CNN14 Backbone (PANNs)
  6x ConvBlock: 1 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048
  BN: shared (D1 frozen, eval mode)
  Conv: frozen (D1 checkpoint)
  FC: Linear(2048 -> 10), trainable
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, nb_tasks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3,3), (1,1), (1,1), bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (3,3), (1,1), (1,1), bias=False)
        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in range(nb_tasks)])

    def forward(self, x, pool_size=(2,2)):
        x = F.relu_(self.bnF[0](self.conv1(x)))
        x = F.relu_(self.bnS[0](self.conv2(x)))
        return F.avg_pool2d(x, pool_size)


class CNN14Backbone(nn.Module):
    def __init__(self, nb_tasks=3):
        super().__init__()
        self.nb_tasks = nb_tasks
        self.spectrogram_extractor = Spectrogram(
            n_fft=c.WIN_SIZE, hop_length=c.HOP_SIZE, win_length=c.WIN_SIZE,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=c.SAMPLE_RATE, n_fft=c.WIN_SIZE, n_mels=c.MEL_BINS,
            fmin=c.FMIN, fmax=c.FMAX, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=30, time_stripes_num=2,
            freq_drop_width=10, freq_stripes_num=2)
        self.bn0 = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(nb_tasks)])
        self.conv_block1 = ConvBlock(1, 64, nb_tasks)
        self.conv_block2 = ConvBlock(64, 128, nb_tasks)
        self.conv_block3 = ConvBlock(128, 256, nb_tasks)
        self.conv_block4 = ConvBlock(256, 512, nb_tasks)
        self.conv_block5 = ConvBlock(512, 1024, nb_tasks)
        self.conv_block6 = ConvBlock(1024, 2048, nb_tasks)
        self.fc = nn.Linear(2048, c.NUM_CLASSES)

    def load_d1_checkpoint(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        own = self.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        self.load_state_dict(own, strict=False)
        print(f"  D1 checkpoint: {loaded}/{len(state)} keys loaded")

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if 'fc' not in name:
                p.requires_grad = False

    def set_bn_eval(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

    def extract_features(self, x, use_spec_aug=False):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[0](x)
        x = x.transpose(1, 3)
        if use_spec_aug and self.training:
            x = self.spec_augmenter(x)
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3,
                  self.conv_block4, self.conv_block5, self.conv_block6]
        for i, block in enumerate(blocks):
            x = block(x)
            x = F.dropout(x, 0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def forward(self, x, use_spec_aug=False):
        feat = self.extract_features(x, use_spec_aug=use_spec_aug)
        return self.fc(feat), feat
