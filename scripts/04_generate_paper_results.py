"""
scripts/04_generate_paper_results.py
Generate all tables and figures for paper submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_comparison_table():
    """Generate LaTeX table for paper"""
    comparison = pd.read_csv('results/method_comparison.csv')
    
    # Format for LaTeX
    latex_table = comparison.to_latex(
        index=False,
        float_format="%.2f",
        caption="Performance comparison of hierarchical agent vs. baselines on UK grid data (June-Nov 2024)",
        label="tab:comparison"
    )
    
    with open('results/paper/table_comparison.tex', 'w') as f:
        f.write(latex_table)
    
    print("✓ LaTeX table generated: results/paper/table_comparison.tex")

def statistical_significance():
    """Perform t-tests between methods"""
    
    # Load daily rewards for each method
    hier = pd.read_csv('results/validation_hierarchical_latest.csv')
    rule = pd.read_csv('results/validation_rule_based.csv')
    
    hier_daily = hier.groupby(hier['datetime'].dt.date)['total_reward'].sum()
    rule_daily = rule.groupby(rule['datetime'].dt.date)['total_reward'].sum()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(hier_daily, rule_daily)
    
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE TEST")
    print(f"{'='*60}")
    print(f"Hierarchical mean: £{hier_daily.mean():.2f} ± £{hier_daily.std():.2f}")
    print(f"Rule-based mean:   £{rule_daily.mean():.2f} ± £{rule_daily.std():.2f}")
    print(f"\nT-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("✓ Difference is statistically significant (p < 0.05)")
    else:
        print("✗ Difference is NOT statistically significant")
    
    return {'t_stat': t_stat, 'p_value': p_value}

if __name__ == "__main__":
    print("Generating paper results...")
    
    # Create output directory
    Path('results/paper').mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_comparison_table()
    
    # Statistical tests
    stats_results = statistical_significance()
    
    print("\n✓ All paper results generated!")
```

---

## **📁 FINAL DIRECTORY STRUCTURE**
```
batteryCLASS/
├── data/
│   ├── raw/                                    # Original downloads
│   │   ├── demanddata_2023.csv
│   │   ├── demanddata_2024.csv
│   │   ├── demanddata_2025.csv
│   │   └── electricitypricesdataset201125.xlsx
│   ├── processed/                              # Combined datasets
│   │   └── uk_battery_dispatch_complete_data.csv
│   └── figures/                                # EDA visualizations
│       └── data_exploration.png
│
├── models/
│   ├── commander/
│   │   ├── best_model.pth                      # Your trained commander
│   │   └── config.json
│   ├── tactician/
│   │   ├── best_model.pth                      # Your trained tactician
│   │   └── config.json
│   └── checkpoints/
│
├── src/
│   ├── __init__.py
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validate_hierarchical.py            # Main validation
│   └── baselines/
│       ├── __init__.py
│       ├── rule_based.py                       # Rule-based baseline
│       └── single_level_rl.py                  # Flattened RL baseline
│
├── scripts/
│   ├── 01_preprocess_data.py                   # Data combination
│   ├── 02_explore_data.py                      # EDA
│   ├── 03_compare_baselines.py                 # Method comparison
│   └── 04_generate_paper_results.py            # Paper tables/figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb               # Interactive EDA
│   ├── 02_results_analysis.ipynb               # Results deep-dive
│   └── 03_visualization.ipynb                  # Custom plots
│
├── results/
│   ├── validation_hierarchical_*.csv           # Your agent results
│   ├── validation_rule_based.csv               # Rule-based results
│   ├── validation_single_level.csv             # Single-level results
│   ├── method_comparison.csv                   # Comparison table
│   ├── metrics_hierarchical.json               # Metrics summary
│   ├── figures/                                # All visualizations
│   │   ├── validation_visualization.png
│   │   ├── method_comparison.png
│   │   └── ...
│   └── paper/                                  # Paper-ready outputs
│       ├── table_comparison.tex
│       └── ...
│
├── README.md
├── requirements.txt
└── .gitignore