"""
src/baselines/single_level_rl.py
Flattened (non-hierarchical) RL agent for comparison
"""

import torch
import torch.nn as nn

class SingleLevelAgent(nn.Module):
    """
    Non-hierarchical baseline: direct state -> action mapping
    No Commander-Tactician decomposition
    """
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.network(state)
    
    def predict(self, state_dict):
        """Match hierarchical agent interface"""
        state = torch.FloatTensor([
            state_dict['hour'] / 24,
            state_dict['price'] / 100,
            state_dict['soc1'],
            state_dict['soc2'],
            state_dict['grid_stress']
        ]).unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(state).cpu().numpy()[0]
        
        return action

# If you trained a single-level baseline, load it here
# Otherwise, this serves as placeholder for comparison