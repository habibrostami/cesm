class TimeEstimator:
    def __init__(self, steps, sub_steps=1):
        self.estimated_step = None
        self.remaining_steps = steps
        self.sub_steps = sub_steps
        self.remaining_sub_steps = sub_steps

    def update(self, time):
        if not self.estimated_step:
            self.estimated_step = time
        elif self.remaining_steps % 5 == 0:
            self.estimated_step = self.estimated_step * 0.6 + time * 0.4
        self.remaining_steps -= 1
        self.remaining_sub_steps = self.sub_steps

    def get_time(self):
        if self.estimated_step is None:
            return None  # or any default value that makes sense
        last_step_time = (self.estimated_step / self.sub_steps) * self.remaining_sub_steps
        return self.estimated_step * (self.remaining_steps - 1) + last_step_time

    def sub_step(self):
        self.remaining_sub_steps -= 1
