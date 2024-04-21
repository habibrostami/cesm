class Loss:
    def __init__(self):
        self.loss_sum = 0.0
        self.loss_count = 0
        self.history = []

    def update(self, loss):
        self.loss_sum += loss
        self.loss_count += 1

    def get_loss(self):
        return self.loss_sum / self.loss_count

    def reset(self):
        self.history.append(self.get_loss())
        self.loss_sum = 0.0
        self.loss_count = 0

    def save(self, path):
        with open(path, 'w') as file:
            for i, v in enumerate(self.history):
                file.write(f'{i},{v}\n')

    def get_best(self):
        if len(self.history) == 0:
            return float('inf')
        return min(self.history)
