import os

class TrackBest:
    def __init__(self, init_value, check_fn, save_path):
        self.check_fn = check_fn
        self.value = init_value
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def update_value(self, value, model):
        if self.check_fn(self.value, value):
            self.value = value
            backbone_path = os.path.join(self.save_path, 'backbone.pth')
            whole_path = os.path.join(self.save_path, 'whole.pth')
            model.save_backbone(backbone_path)
            model.save_all(whole_path)
