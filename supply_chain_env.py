import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SupplyChainEnv(gym.Env):
    def __init__(self):
        super(SupplyChainEnv, self).__init__()
        self.action_space = spaces.Discrete(4) # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
        self.goal_pos = np.array([8, 8]) # The delivery destination
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # [x, y, fuel]
        self.state = np.array([0, 0, 100], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        x, y, fuel = self.state
        reward = -1  # Small penalty for every move (encourages speed)
        terminated = False
        
        # 1. Update Position & Check Walls
        if action == 0: # Up
            if y < 9: y += 1
            else: reward -= 5 # Wall Penalty!
        elif action == 1: # Down
            if y > 0: y -= 1
            else: reward -= 5
        elif action == 2: # Left
            if x > 0: x -= 1
            else: reward -= 5
        elif action == 3: # Right
            if x < 9: x += 1
            else: reward -= 5

        # 2. Consume Fuel
        fuel -= 1
        
        # 3. Check if reached Goal
        if x == self.goal_pos[0] and y == self.goal_pos[1]:
            reward += 100 # BIG WIN!
            terminated = True
            
        # 4. Check if Out of Fuel
        if fuel <= 0:
            reward -= 50 # Failure Penalty
            terminated = True

        self.state = np.array([x, y, fuel], dtype=np.float32)
        return self.state, reward, terminated, False, {}
    
