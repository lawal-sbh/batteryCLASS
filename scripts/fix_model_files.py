"""
scripts/fix_model_files.py
Copy latest trained models to best_model.pth for validation
"""

import shutil
from pathlib import Path

print("="*60)
print("FIXING MODEL FILES")
print("="*60)

# Source files (your actual trained models)
commander_source = 'models/commander/commander_episode_1000_20251122_171029.pth'
tactician_source = 'models/tactician/tactician_episode_1000_20251122_171029.pth'

# Destination files (what validation script expects)
commander_dest = 'models/commander/best_model.pth'
tactician_dest = 'models/tactician/best_model.pth'

# Copy files
print("\nCopying trained models...")

if Path(commander_source).exists():
    shutil.copy(commander_source, commander_dest)
    size = Path(commander_dest).stat().st_size
    print(f"✓ Commander: {size:,} bytes copied to best_model.pth")
else:
    print(f"✗ Commander source not found: {commander_source}")

if Path(tactician_source).exists():
    shutil.copy(tactician_source, tactician_dest)
    size = Path(tactician_dest).stat().st_size
    print(f"✓ Tactician: {size:,} bytes copied to best_model.pth")
else:
    print(f"✗ Tactician source not found: {tactician_source}")

print("\n" + "="*60)
print("✓✓✓ MODEL FILES FIXED! ✓✓✓")
print("="*60)
print("\nYou can now run validation:")
print("  python src/validation/validate_hierarchical.py")