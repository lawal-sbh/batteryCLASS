"""
scripts/test_model_loading.py
Test loading your trained models and inspect their structure
"""

import torch
from pathlib import Path

def inspect_checkpoint(path):
    """Load and inspect a checkpoint"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {Path(path).name}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        print("\n📦 Checkpoint Contents:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"  - {key}: dict with {len(value)} items")
            elif isinstance(value, torch.Tensor):
                print(f"  - {key}: Tensor {value.shape}")
            else:
                print(f"  - {key}: {type(value).__name__} = {value}")
        
        # If it has policy_state_dict, show its structure
        if 'policy_state_dict' in checkpoint:
            print("\n🎯 Policy State Dict Keys:")
            for key in checkpoint['policy_state_dict'].keys():
                shape = checkpoint['policy_state_dict'][key].shape
                print(f"  - {key}: {shape}")
    else:
        print(f"\n⚠️ Not a dict, it's a: {type(checkpoint)}")

# Test both models
print("="*60)
print("MODEL CHECKPOINT INSPECTION")
print("="*60)

commander_path = 'models/commander/best_model.pth'
tactician_path = 'models/tactician/best_model.pth'

if Path(commander_path).exists() and Path(commander_path).stat().st_size > 0:
    inspect_checkpoint(commander_path)
else:
    print(f"\n⚠️ {commander_path} not found or empty")

if Path(tactician_path).exists() and Path(tactician_path).stat().st_size > 0:
    inspect_checkpoint(tactician_path)
else:
    print(f"\n⚠️ {tactician_path} not found or empty")

print("\n" + "="*60)
print("✓ Inspection complete!")
print("="*60)