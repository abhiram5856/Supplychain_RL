import random
import numpy as np
from collections import deque

class ReplayMemory:
    def __init__(self, capacity=2000):
        # We store up to 2000 'experiences'
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Returns a random batch of memories to learn from."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    # Test the memory
    mem = ReplayMemory(capacity=10)
    mem.push([0,0,100], 3, -1, [1,0,99], False)
    print(f"Memory length: {len(mem)}")
    print(f"Sampled one memory: {mem.sample(1)}")