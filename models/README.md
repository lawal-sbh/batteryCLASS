
# Model Directory Structure

## File Organization
- `commander/`: Day-ahead planning agents
- `tactician/`: Real-time execution agents  
- `checkpoints/`: Training snapshots

## Model Format
- Serialized PyTorch models (.pkl)
- Includes policy weights, optimizer states, hyperparameters
- Compatible with custom loading utilities

## Training Protocol
1. Commander trains on 24-hour episodes
2. Tactician trains on 30-minute intervals
3. Settlement engine coordinates learning feedback
        