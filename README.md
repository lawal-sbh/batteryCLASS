# batteryCLASS — Hierarchical Multi-Objective RL for UK Grid-Scale BESS Dispatch

> **MSc Renewable Energy · Cranfield University**  
> Optimising battery energy storage dispatch across UK electricity markets using Reinforcement Learning

---

## The Problem

Grid-scale battery storage assets in the UK must make real-time dispatch decisions across multiple revenue streams simultaneously — wholesale day-ahead trading, Balancing Mechanism (BM) actions, and ancillary service markets (FFR, DC, DR). A purely price-chasing strategy leaves significant revenue on the table and can actively destabilise the grid during periods of high renewable penetration.

This project builds a **Hierarchical Multi-Objective Reinforcement Learning** framework that learns to optimise BESS dispatch while explicitly balancing profit maximisation with grid stability contribution.

---

## Key Results

| Strategy | Revenue (simulated period) | Notes |
|----------|---------------------------|-------|
| Heuristic (rule-based) | £4,693 | Peak/off-peak threshold logic |
| LP Optimal (benchmark) | £11,700 | Perfect foresight upper bound |
| **RL Agent (in training)** | **Target: close gap to LP** | Multi-objective reward |

The £7,007 performance gap between heuristic and optimal represents the core research problem. The RL agent is trained to close this gap without the benefit of perfect price foresight.

---

## Architecture

```
batteryCLASS/
├── src/
│   ├── battery_model.py        # Physical BESS simulation (SoC, degradation, C-rate limits)
│   ├── rl_environment.py       # Gymnasium-compatible RL training environment
│   └── simple_data_loader.py   # UK market data interface (Elexon / National Grid ESO)
├── notebooks/
│   ├── testing.py              # Unit tests and component validation
│   ├── trading_simulation.py   # Heuristic & LP benchmark evaluation
│   └── smarter_trading.py      # Advanced heuristic strategies
├── ai_vs_human_performance.png # Performance comparison chart
└── requirements.txt
```

---

## Physical Battery Model

The `BatteryModel` class simulates a realistic grid-scale BESS with:
- State of Charge (SoC) constraints with configurable min/max bounds
- Round-trip efficiency losses on charge and discharge cycles
- Degradation modelling based on cycle depth
- C-rate limits (maximum charge/discharge power relative to capacity)

---

## RL Environment

Built on the **Gymnasium** interface for compatibility with standard RL libraries (Stable-Baselines3, RLlib). The environment:
- Observation space: current SoC, half-hourly price forecast, time features, grid frequency signal
- Action space: continuous charge/discharge power command
- Reward function: weighted combination of market revenue + grid stability contribution penalty term
- Episode structure: rolling 24-hour windows on historical UK market data

---

## Market Context

The UK electricity market context this model operates in:
- **Half-hourly settlement periods** (Elexon BSC)
- **Balancing Mechanism** — real-time system balancing by National Grid ESO
- **Ancillary services** — Dynamic Containment (DC), Dynamic Regulation (DR), Dynamic Moderation (DM)
- **Frequency response** — primary and secondary services increasingly critical as synchronous generation retires

This work connects directly to the UK's [Stability Pathfinder](https://www.nationalgrideso.com/future-energy/projects/stability-pathfinder) programme, which is procuring grid-forming capability from storage assets to replace retiring synchronous plant.

---

## Installation & Usage

```bash
git clone https://github.com/lawal-sbh/batteryCLASS.git
cd batteryCLASS
pip install -r requirements.txt
```

Run the benchmark comparison:
```bash
python notebooks/trading_simulation.py
```

Train the RL agent (in development):
```bash
python src/rl_environment.py --train --episodes 1000
```

---

## Status & Roadmap

- ✅ Physical battery simulation with realistic constraints
- ✅ UK market data integration
- ✅ Multiple trading strategies benchmarked
- ✅ Gymnasium RL environment ready for agent training
- ⏳ RL agent training and evaluation (in progress)
- ⏳ Multi-objective reward function refinement
- ⏳ Integration with real-time Elexon API data

---

## Related Work

This project sits alongside my MSc dissertation on **Grid-Forming Control for Hybrid PEM-Alkaline Electrolysers** and group project on **AI-Assisted SOEC Control** — together forming a portfolio on AI-enabled control of grid-interactive energy assets.

---

**Author:** Hassan Lawal · Cranfield University · [LinkedIn](https://linkedin.com/in/hassanlawal-13943475)  
**Stack:** Python · NumPy · pandas · SciPy · Gymnasium · scikit-learn
