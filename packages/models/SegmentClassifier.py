import torch
from torch import nn
from packages.models.UNet import UNet

from packages.models.Resnet import get_model


class SegmentClassifier(nn.Module):
    def __init__(self, segment_model=None, res1=None, res2=None, res3=None, num_classes=2):
        super().__init__()

        if not segment_model:
            segment_model = UNet(3, 2, True)


        if not res1:
            res1 = get_model(18, only_backbone=True)
        if not res2:
            res2 = get_model(18, only_backbone=True)
        if not res3:
            res3 = get_model(18, only_backbone=True)

        self.segment_model = segment_model
        self.res1 = res1
        self.res2 = res2
        self.res3 = res3

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8 * 3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

        self.segment_model.freeze()

    def forward(self, img):
        mask = self.segment_model(img)
        mask = torch.where(mask[:, 0:1, :, :] > mask[:, 1:2, :, :], 0.0, 1.0).repeat(1, 3, 1, 1)

        x1 = self.res1(mask)
        x1 = torch.flatten(x1, 1)

        x2 = self.res2(img)
        x2 = torch.flatten(x2, 1)

        x3 = self.res3(mask * img)
        x3 = torch.flatten(x3, 1)

        out = self.classifier(torch.cat((x1, x2, x3), 1))
        self.features = out
        logits = self.fc(out)

        return logits

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def to(self, *args, **kwargs):
        if self.segment_model:
            self.segment_model.to(*args, **kwargs)
            self.res1.to(*args, **kwargs)
            self.res2.to(*args, **kwargs)
            self.res3.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def save_backbone(self, path):
        pass

    def save_all(self, path):
        torch.save(self.state_dict(), path)

    def load_backbone(self, path):
        pass

    def load_all(self, path):
        self.load_state_dict(torch.load(path))
