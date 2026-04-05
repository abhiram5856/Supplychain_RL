import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNBrain(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNBrain, self).__init__()
        # Layer 1: Takes the 3 state numbers (x, y, fuel)
        self.fc1 = nn.Linear(state_size, 64)
        # Layer 2: A hidden layer to help the AI learn patterns
        self.fc2 = nn.Linear(64, 64)
        # Layer 3: Outputs 4 numbers (one for each possible action)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        # We use ReLU activation to help the network learn complex rules
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == "__main__":
    # Test the brain with a fake state
    model = DQNBrain(3, 4)
    test_input = torch.tensor([0.0, 0.0, 100.0]) 
    output = model(test_input)
    print(f"Brain's raw thoughts (Q-Values) for [Up, Down, Left, Right]:\n{output.detach().numpy()}")