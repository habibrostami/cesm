import torch
from torch import nn
from transformers import ResNetModel
from transformers import SwinModel


class Resnet(nn.Module):
    def __init__(self, model_id="microsoft/resnet-18", num_labels=2):
        super(Resnet, self).__init__()
        self.K = 512
        self.res = ResNetModel.from_pretrained(model_id)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(self.res.config.hidden_sizes[-1], self.K)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.K, num_labels))
        self.num_labels = num_labels

    def forward(self, img):
        outputs = self.res(pixel_values=img)

        flat = self.flat(outputs[1])
        logits = self.classifier(flat)
        return logits


class SwinClassifier(nn.Module):
    def __init__(self, model_id="microsoft/swin-tiny-patch4-window7-224",
                 num_labels=2):  # microsoft/swin-tiny-patch4-window7-224
        super(SwinClassifier, self).__init__()
        self.model_id = model_id
        self.swin = SwinModel.from_pretrained(model_id)
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.swin.num_features, num_labels)

    def forward(self, img):
        outputs = self.swin(pixel_values=img)
        logits = self.classifier(outputs[1])

        return logits


class Encoder(nn.Module):
    def __init__(self, num_labels=2):
        super(Encoder, self).__init__()

        self.trans_core = SwinClassifier(num_labels=num_labels).swin
        self.cnn_core = Resnet(num_labels=num_labels).res

        self.len_res = self.cnn_core.config.hidden_sizes[-1]
        self.len_trans = self.trans_core.num_features
        self.K = self.len_trans * 3

        self.cnn_to_trans = nn.Linear(self.len_res, self.len_trans)
        self.attention_layer = nn.MultiheadAttention(self.len_trans, num_heads=6)

    def forward(self, img):
        trans_out = self.trans_core(pixel_values=img)[1]
        cnn_out = self.cnn_core(pixel_values=img)[1]
        cnn_out = torch.flatten(cnn_out, start_dim=1)
        cnn_out = self.cnn_to_trans(cnn_out)
        cat_res = self.attention_layer(trans_out, cnn_out, cnn_out)
        cat_res = torch.cat((cat_res[0], trans_out, cnn_out), dim=1)

        return cat_res


class JointModel(nn.Module):
    def __init__(self, num_labels=2):
        super(JointModel, self).__init__()

        self.num_labels = num_labels

        self.encoder = Encoder(num_labels=num_labels)
        self.features = None

        self.merge_layer = nn.Sequential(nn.Linear(self.encoder.len_res + self.encoder.len_trans, self.encoder.K))
        self.fc = nn.Linear(self.encoder.K, num_labels)

    def forward(self, img):
        cat_res = self.encoder(img)
        logits = self.fc(cat_res)
        self.features = cat_res
        # print(logits)


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
