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


###
# class Loss:
#     def __init__(self):
#         self.total_loss = 0
#         self.count = 0
#
#     def update(self, loss_value, n=1):
#         self.total_loss += loss_value * n
#         self.count += n
#
#     def reset(self):
#         self.total_loss = 0
#         self.count = 0
#
#     def get_value(self):
#         return self.total_loss / self.count if self.count != 0 else 0