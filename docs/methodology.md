# Research Methodology
**MSc Thesis - Hierarchical Multi-Objective RL for UK Battery Dispatch**

## Statistical Validation Framework

### Experimental Design
- **Sample Size**: 5 random seeds for robustness testing
- **Baseline**: Established heuristic strategy (£4,693 daily revenue)
- **Statistical Test**: One-sample t-test against fixed baseline
- **Confidence Level**: 95% confidence intervals
- **Effect Size**: Cohen's d for practical significance interpretation

### Performance Metrics
- **Primary**: Daily revenue (£)
- **Secondary**: Statistical significance (p-value)
- **Tertiary**: Effect size (Cohen's d)

## Reinforcement Learning Setup

### Environment Specification
```python
State Space: [SOC_t, P_t/100, h_t/24] ∈ [0,1]³
Action Space: {HOLD, CHARGE, DISCHARGE} 
Reward Function: r_t = revenue_t/1000 - penalty_invalid