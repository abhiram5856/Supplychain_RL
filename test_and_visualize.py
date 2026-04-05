import torch
import numpy as np
import matplotlib.pyplot as plt
from supply_chain_env import SupplyChainEnv
from dqn_model import DQNBrain

# 1. Load the Environment and the Brain
env = SupplyChainEnv()
model = DQNBrain(3, 4)
model.load_state_dict(torch.load("supply_chain_model.pth"))
model.eval() # Set to evaluation mode

state, _ = env.reset()
path = [state[:2].copy()] # Track the (x, y) coordinates
total_reward = 0
done = False

print("🚚 The trained truck is starting its delivery...")

# 2. Let the AI drive (No randomness/Epsilon here!)
while not done:
    with torch.no_grad():
        state_v = torch.tensor(state, dtype=torch.float32)
        action = model(state_v).argmax().item()
    
    state, reward, done, _, _ = env.step(action)
    path.append(state[:2].copy())
    total_reward += reward
    if len(path) > 50: break # Safety break

# 3. Plot the Results
path = np.array(path)
plt.figure(figsize=(8,8))
plt.plot(path[:, 0], path[:, 1], marker='o', color='blue', label='AI Path')
plt.scatter([0], [0], color='green', s=100, label='Warehouse (Start)')
plt.scatter([8], [8], color='red', s=100, label='Customer (Goal)')

plt.grid(True)
plt.xlim(-1, 10)
plt.ylim(-1, 10)
plt.title(f"Supply Chain AI Route (Total Reward: {total_reward})")
plt.legend()
plt.show()