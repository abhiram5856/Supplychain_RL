import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from supply_chain_env import SupplyChainEnv
from dqn_model import DQNBrain
from memory import ReplayMemory

# --- Hyperparameters ---
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.997
LR = 0.001

# 1. Setup Environment and Brain
env = SupplyChainEnv()
policy_net = DQNBrain(3, 4)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(5000)
criterion = nn.MSELoss()

epsilon = EPS_START

print("🚀 Training Started... If it's working, you will see Episode 0 below soon.")

for episode in range(800):
    state, _ = env.reset()
    total_reward = 0
    
    for t in range(100): # Max 100 steps per delivery
        # --- EXPLORE OR EXPLOIT ---
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_v = torch.tensor(state, dtype=torch.float32)
                action = policy_net(state_v).argmax().item()

        # --- STEP THE WORLD ---
        next_state, reward, done, _, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward

        # --- ACTUAL LEARNING ---
        if len(memory) > BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            
            # Prepare batch data
            b_state = torch.tensor([t[0] for t in transitions], dtype=torch.float32)
            b_action = torch.tensor([t[1] for t in transitions]).unsqueeze(1)
            b_reward = torch.tensor([t[2] for t in transitions], dtype=torch.float32)
            b_next_state = torch.tensor([t[3] for t in transitions], dtype=torch.float32)
            b_done = torch.tensor([t[4] for t in transitions], dtype=torch.float32)

            # Get current Q values from brain
            current_q = policy_net(b_state).gather(1, b_action)
            
            # Get maximum Q values for next state
            with torch.no_grad():
                next_q = policy_net(b_next_state).max(1)[0]
                expected_q = b_reward + (GAMMA * next_q * (1 - b_done))

            # Backpropagation (Teaching the brain)
            loss = criterion(current_q.squeeze(), expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break
            
    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    
    # Print EVERY episode so you can see it's working
    print(f"Episode {episode} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f}")

print("✅ DONE! Check your terminal for the reward history.")

# Save the model weights
torch.save(policy_net.state_dict(), "supply_chain_model.pth")
print("💾 Model saved as 'supply_chain_model.pth'!")