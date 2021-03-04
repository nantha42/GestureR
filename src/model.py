from torch import nn
from icecream import ic
import torch
import numpy as np
from icecream import ic


class Model(nn.Module):
    def __init__(self, nclasses):
        super(Model, self).__init__()
        self.m = nn.Conv3d(1, 20, 3, stride=1)
        self.p = nn.MaxPool3d((3, 3, 3), stride=2)

        self.m1 = nn.Conv3d(20, 30, 3, stride=3)
        self.p1 = nn.MaxPool3d((3, 3, 3), stride=1)
        self.f_ch = 64
        self.m2 = nn.Conv3d(30, self.f_ch, 1, stride=1)
        self.p2 = nn.MaxPool3d((4, 3, 3), stride=1)

        encod = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.encoder = nn.TransformerEncoder(encod, num_layers=2)
        # if video length changed, the value below needs to be
        # changed

        self.f1 = nn.Linear(self.f_ch * 256, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, nclasses)

    def forward(self, x):
        r = self.m(x)
        r = self.p(r)

        # ic(r.shape)
        r = self.m1(r)
        r = self.p1(r)
        # ic(r.shape)

        r = self.m2(r)
        r = self.p2(r)

        # ic(r.shape)
        # print(r.shape)
        assert torch.numel(r) % (x.shape[0] * self.f_ch) == 0

        r = r.reshape((-1, self.f_ch, torch.numel(r) // (x.shape[0] * self.f_ch)))
        r = self.encoder(r)
        # ic("Enc out",r.shape)
        r = r.reshape(x.shape[0], -1)

        r = nn.functional.relu(self.f1(r))
        r = nn.functional.relu(self.f2(r))
        r = nn.functional.softmax(self.f3(r), dim=-1)
        return r

