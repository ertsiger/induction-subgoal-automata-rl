import collections
import numpy as np

Experience = collections.namedtuple("Experience", field_names=["state", "action", "next_state", "is_terminal",
                                                               "observations", "observations_changed"])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[x] for x in indices]
