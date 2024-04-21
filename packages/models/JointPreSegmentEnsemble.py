import torch
from torch import nn

from packages.models.Joint import Encoder as JointModel


class JointPreSegmentEnsemble(nn.Module):
    def __init__(self, cm_cc, cm_mlo, dm_cc, dm_mlo, num_labels=2):
        super(JointPreSegmentEnsemble, self).__init__()

        cm_cc.freeze()
        cm_mlo.freeze()
        dm_cc.freeze()
        dm_mlo.freeze()

        self.cm_cc = cm_cc
        self.cm_mlo = cm_mlo
        self.dm_cc = dm_cc
        self.dm_mlo = dm_mlo

        self.K = cm_cc.K * 4

        self.fc = nn.Linear(self.K, num_labels)

    def forward(self, x1, x2, x3, x4, m1, m2, m3, m4):
        self.cm_cc(x1, m1)
        self.cm_mlo(x2, m2)
        self.dm_cc(x3, m3)
        self.dm_mlo(x4, m4)

        x1 = self.cm_cc.features
        x2 = self.cm_mlo.features
        x3 = self.dm_cc.features
        x4 = self.dm_mlo.features

        x = torch.cat((x1, x2, x3, x4), dim=1)
        return self.fc(x)

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
