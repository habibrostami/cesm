import torch


class ConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy_hist = []
        self.sensitivity_hist = []
        self.specificity_hist = []
        self.dice_hist = []

    def update(self, predicted, label):
        predicted = predicted.view(-1)
        label = label.view(-1)

        tp = torch.sum((predicted == label).type(torch.LongTensor) * (label == 1).type(torch.LongTensor)).item()
        tn = torch.sum((predicted == label).type(torch.LongTensor) * (label == 0).type(torch.LongTensor)).item()
        fp = torch.sum((predicted != label).type(torch.LongTensor) * (predicted == 1).type(torch.LongTensor)).item()
        fn = torch.sum((predicted != label).type(torch.LongTensor) * (predicted == 0).type(torch.LongTensor)).item()

        assert tp + tn + fp + fn == len(predicted), f"{tp + tn + fp + fn} != {len(predicted)}"

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def get_value(self, metric):
        if metric == "accuracy":
            if self.tp + self.fp + self.tn + self.fn == 0:
                return 0
            return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) * 100
        if metric == "sensitivity":
            if self.tp + self.fn == 0:
                return 0
            return self.tp / (self.tp + self.fn) * 100
        if metric == "specificity":
            if self.tn + self.fp == 0:
                return 0
            return self.tn / (self.tn + self.fp) * 100
        if metric == "dice":
            if self.tp + self.fp + self.fn == 0:
                return 0
            return (2 * self.tp) / (2 * self.tp + self.fp + self.fn) * 100

    def reset(self):
        self.accuracy_hist.append(self.get_value("accuracy"))
        self.sensitivity_hist.append(self.get_value("sensitivity"))
        self.specificity_hist.append(self.get_value("specificity"))
        self.dice_hist.append(self.get_value("dice"))

        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def get_best(self, metric='sensitivity'):
        if metric == "accuracy":
            hist = self.accuracy_hist
        elif metric == 'sensitivity':
            hist = self.sensitivity_hist
        elif metric == 'specificity':
            hist = self.specificity_hist
        elif metric == 'dice':
            hist = self.dice_hist
        else:
            hist = []
        if len(hist) == 0:
            return 0
        return max(hist)

    def save(self, path):
        with open(path + "_accuracy.csv", 'w') as file:
            for i, v in enumerate(self.accuracy_hist):
                file.write(f'{i},{v}\n')
        with open(path + "_sensitivity.csv", 'w') as file:
            for i, v in enumerate(self.sensitivity_hist):
                file.write(f'{i},{v}\n')
        with open(path + "_specificity.csv", 'w') as file:
            for i, v in enumerate(self.specificity_hist):
                file.write(f'{i},{v}\n')
        with open(path + "_dice.csv", 'w') as file:
            for i, v in enumerate(self.dice_hist):
                file.write(f'{i},{v}\n')
