class Sensitivity:
    def __init__(self):
        self.sensitivity_sum = 0.0
        self.sensitivity_count = 0
        self.history = []

    def update(self, true_positives, actual_positives):
        sensitivity = true_positives / (actual_positives + 1e-6)
        self.sensitivity_sum += sensitivity
        self.sensitivity_count += 1

    def get_sensitivity(self):
        if self.sensitivity_count == 0:
            return 0
        return self.sensitivity_sum / self.sensitivity_count

    def reset(self):
        self.history.append(self.get_sensitivity())
        self.sensitivity_sum = 0.0
        self.sensitivity_count = 0

    def save(self, path):
        with open(path, 'w') as file:
            for i, v in enumerate(self.history):
                file.write(f'{i},{v}\n')

    def get_best(self):
        if len(self.history) == 0:
            return 0
        return max(self.history)
