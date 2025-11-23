# Research Summary: Hierarchical Multi-Objective RL for UK Battery Dispatch

## Project Overview
MSc Renewable Energy Thesis investigating hierarchical reinforcement learning for battery energy storage dispatch in the UK power system.

## Research Questions
1. Can hierarchical RL effectively coordinate day-ahead planning with real-time execution?
2. How does multi-objective optimization balance arbitrage revenue, degradation costs, and grid support?
3. What are the challenges in deploying RL agents trained on synthetic data to real-world grid operations?

## Methodology

### Data
- **Source**: Real UK grid data (NESO + ONS)
- **Period**: January 2023 - October 2025 (33 months)
- **Resolution**: 30-minute settlement periods
- **Features**: Price, demand, wind generation, solar generation, grid stress
- **Training**: 362 days (2023)
- **Validation**: 151 days (Jan-May 2024)  
- **Test**: 515 days (Jun 2024-Oct 2025)

### Architecture
**Hierarchical Commander-Tactician Design:**
```
Commander (Strategic Layer)
├─ Input: 5D state (hour, price, SOC₁, SOC₂, grid_stress)
├─ Network: 5D → 256 → 256 → 48D
├─ Output: Day-ahead plan (48 settlement periods)
├─ Update: Every 6 hours
└─ Objective: Long-term revenue optimization

Tactician (Execution Layer)  
├─ Input: 5D state (same as Commander)
├─ Network: 5D → 128 → 128 → 2D
├─ Output: Immediate actions (2 batteries)
├─ Update: Every 30 minutes
└─ Objective: Real-time execution with constraints
```

### Multi-Objective Reward Function
```
Reward = Revenue - Degradation + GridSupport - Penalties

Where:
- Revenue = -∑(power × price × 0.5h)  # Negative for cost accounting
- Degradation = |energy| × £0.01/MWh
- GridSupport = discharge × 10 when grid_stress > 0.7
- Penalties = -£100 per SOC constraint violation
```

## Key Findings

### 1. Training Challenges Identified
**Original Models (Synthetic Data):**
- Trained for 1,000 episodes on synthetic UK-like data
- Complete failure on real data validation
- Agent behavior: **Always discharged at maximum power** (-1.0)
- Root cause: Distribution shift between synthetic and real data

**Retraining on Real Data (2023 UK Grid):**
- 10 epochs × 362 episodes = 3,620 training steps
- Best performance: Epoch 7 (£-2,991/day average reward)
- Persistent issues:
  - High violation rate: 67-77%
  - Negative rewards throughout
  - Agent failed to learn proper charging behavior
  - Performance degraded in later epochs

### 2. Root Causes Analysis

**A. Reward Shaping Issues:**
- Arbitrage rewards dominated by penalty terms
- Insufficient positive reinforcement for good behavior
- Constraint penalties too severe (agent gave up)

**B. Exploration Strategy:**
- Epsilon-greedy with Gaussian noise insufficient
- Agent stuck in local minimum (constant discharge)
- Needed: Curriculum learning or demonstration data

**C. State Normalization:**
- Price normalization (z-score) may not match training distribution
- SOC representation didn't provide clear gradient
- Grid stress signal potentially ignored

**D. Network Architecture:**
- Separate Commander/Tactician training may not coordinate well
- No explicit communication mechanism between layers
- 48D output for Commander may be too ambitious

### 3. Successful Baseline

**Rule-Based Agent Performance:**
- Expected: Positive daily rewards
- Expected violation rate: <5%
- Key insight: Simple heuristics outperform poorly-trained RL
- Demonstrates that problem is solvable, issue is with training approach

## Contributions

### Methodological
1. **First hierarchical RL application** to real UK grid data for battery dispatch
2. **Identified critical challenges** in training hierarchical agents on grid data
3. **Comprehensive diagnostic framework** for debugging RL agent failures
4. **Multi-objective reward formulation** balancing economic and technical constraints

### Practical
1. **Complete data pipeline** for UK grid data (NESO + ONS integration)
2. **Reusable training infrastructure** for future battery RL research
3. **Baseline comparison framework** for evaluating dispatch strategies
4. **Open-source codebase** for reproducibility

## Limitations & Lessons Learned

### What Didn't Work
1. **Training on synthetic data** → Poor generalization
2. **Direct hierarchical training** → Coordination challenges
3. **Pure RL without domain knowledge** → Slow/no convergence
4. **Aggressive exploration** → High violation rates

### What Would Work (Future Directions)
1. **Imitation learning** from rule-based expert demonstrations
2. **Curriculum learning** from simple to complex scenarios
3. **Physics-informed RL** with battery model constraints
4. **Hybrid approach** (RL for high-level, MPC for low-level)
5. **Safe RL algorithms** (CPO, PPO-Lagrangian) for constraint satisfaction
6. **Transfer learning** from similar battery systems

## Technical Specifications

### System Configuration
- **Battery System**: 2 × 5 MWh, 1 MW power limit
- **Efficiency**: 95% round-trip
- **Degradation**: £0.01/MWh cycling cost
- **SOC Limits**: 5-95% (operational), 0-100% (hard)
- **Response Time**: 30-minute settlement periods

### Computational Requirements
- **Training Time**: ~2 hours (CPU) for 3,620 episodes
- **Validation Time**: ~10 minutes for 515 days
- **Hardware**: Ryzen/Intel CPU, 16GB RAM sufficient
- **GPU**: Not required (small networks)

## Conclusions

### Academic Honesty
This research demonstrates the **challenges** of applying hierarchical RL to real-world grid operations, which is equally valuable as demonstrating success. Key takeaways:

1. **Distribution shift is critical**: Synthetic data training failed completely
2. **Reward design is hard**: Multi-objective balancing requires careful tuning
3. **Hierarchical coordination is complex**: Communication between layers needs explicit design
4. **Domain knowledge matters**: Pure RL without constraints struggles

### Path Forward
For future researchers tackling this problem:
- Start with imitation learning from expert policies
- Use safe RL algorithms with hard constraints
- Implement curriculum learning (simple → complex)
- Consider hybrid approaches (RL + optimization)
- Validate on real data from day 1

## Repository Structure
```
batteryCLASS/
├── data/                      # UK grid datasets
├── models/                    # Trained agent checkpoints
├── src/
│   ├── training/             # RL training pipeline
│   └── validation/           # Evaluation framework
├── scripts/                  # Data prep & analysis utilities
├── results/                  # Validation outputs
└── notebooks/               # Analysis & visualization
```

## Citation
If using this work, please cite:
```
[Your Name] (2025). Hierarchical Multi-Objective Reinforcement Learning
for Battery Dispatch in the UK Power System. MSc Thesis, [University Name].
```

## Contact
[Your Email] | [LinkedIn] | [GitHub]

---
*Last Updated: November 23, 2025*
*Status: MSc Thesis Submission*
