# batteryCLASS
Optimization
# Hierarchical Multi-Objective RL for UK Battery Dispatch

**MSc Thesis Project - Cranfield University**  
*Renewable Energy MSc*

## Project Overview

This research develops a Hierarchical Multi-Objective Reinforcement Learning framework for optimal dispatch of grid-scale battery storage in the UK electricity market.

### Key Components

- **Physical Battery Model**: Realistic simulation of grid-scale BESS with degradation
- **UK Market Interface**: Integration with Elexon/National Grid data structures  
- **RL Trading Environment**: Gym-compatible environment for AI agent training
- **Multi-Objective Optimization**: Balancing profit maximization with grid stability

### Project Structure
batteryCLASS/
├── src/ # Source code
│ ├── battery_model.py # Physical battery simulation
│ ├── rl_environment.py # RL training environment
│ └── simple_data_loader.py # Market data interface
├── notebooks/ # Experiments & analysis
│ ├── testing.py # Basic functionality tests
│ ├── trading_simulation.py # Trading strategy evaluation
│ └── smarter_trading.py # Advanced heuristic strategies
├── docs/ # Documentation
└── data/ # Market data (git-ignored)

### Current Capabilities

✅ **Physical battery simulation** with realistic constraints  
✅ **Market data integration** with UK price structures  
✅ **Multiple trading strategies** implemented and benchmarked  
✅ **RL environment** ready for agent training  
✅ **Performance gap identified**: £4,693 (heuristic) vs £11,700 (optimal)

### Next Steps

- [ ] Train and evaluate RL agents
- [ ] Integrate real historical market data
- [ ] Implement multi-objective reward functions
- [ ] Develop hierarchical control architecture

## Installation

```bash
git clone [your-repo-url]
cd batteryCLASS
pip install -r requirements.txt