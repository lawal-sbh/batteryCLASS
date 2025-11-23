# Run this diagnostic script: scripts/check_models.py

from pathlib import Path
import os

print("="*60)
print("MODEL FILE DIAGNOSTICS")
print("="*60)

model_paths = [
    'models/commander/best_model.pth',
    'models/tactician/best_model.pth'
]

for path in model_paths:
    p = Path(path)
    if p.exists():
        size = os.path.getsize(p)
        print(f"\n✓ Found: {path}")
        print(f"  Size: {size:,} bytes ({size/1024:.1f} KB)")
        if size == 0:
            print(f"  ⚠️ WARNING: File is EMPTY!")
    else:
        print(f"\n✗ NOT FOUND: {path}")

# Check what files actually exist in models/
print("\n" + "="*60)
print("FILES IN models/ DIRECTORY:")
print("="*60)

models_dir = Path('models')
if models_dir.exists():
    for item in models_dir.rglob('*'):
        if item.is_file():
            size = os.path.getsize(item)
            print(f"  {item} ({size:,} bytes)")
else:
    print("  ⚠️ models/ directory doesn't exist!")