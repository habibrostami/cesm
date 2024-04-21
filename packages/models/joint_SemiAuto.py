import warnings

import torch
from torch import nn

from packages.Dataset import SegmentDataset
from packages.models.Joint import Encoder as JointModel


class JointSemiAuto(nn.Module):
    def __init__(self, num_labels=2):
        super(JointSemiAuto, self).__init__()


        # self.mask = mask_dir
        self.classifier1 = JointModel(num_labels=num_labels)
        self.classifier2 = JointModel(num_labels=num_labels)
        self.classifier3 = JointModel(num_labels=num_labels)

        self.features = None
        self.K = self.classifier1.K * 3

        self.fc = nn.Linear(self.K, num_labels)

    def forward(self, img):
        # print(mask)
        # mask = self.segmentor(img)[:, 1].unsqueeze(1)
        # print(img.shape)
        mask = 0  #mask
        mask = mask.repeat(1, 3, 1, 1)

        x1 = self.classifier1(mask)
        x2 = self.classifier2(mask * img)
        x3 = self.classifier3(img)

        x = torch.cat((x1, x2, x3), dim=1)
        self.features = x
        logits = self.fc(x)


        return logits

    def save_backbone(self, path):
        pass

    def save_all(self, path):
        torch.save(self.state_dict(), path)

    def load_backbone(self, path):
        pass

    def load_all(self, path):
        self.load_state_dict(torch.load(path))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
