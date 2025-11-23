"""
scripts/03_compare_baselines.py
Compare hierarchical agent against all baselines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_all_results():
    """Load results from all methods"""
    results_dir = Path('results')
    
    methods = {
        'Hierarchical': results_dir / 'validation_hierarchical_*.csv',
        'Rule-Based': results_dir / 'validation_rule_based.csv',
        'Single-Level': results_dir / 'validation_single_level.csv',
        'MPC': results_dir / 'validation_mpc.csv'
    }
    
    # Load each
    all_results = {}
    for name, pattern in methods.items():
        try:
            files = list(results_dir.glob(pattern.name))
            if files:
                df = pd.read_csv(files[0])  # Latest file
                all_results[name] = df
                print(f"✓ Loaded {name}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Could not load {name}: {e}")
    
    return all_results

def compare_metrics(all_results):
    """Statistical comparison of methods"""
    
    comparison = []
    
    for method, df in all_results.items():
        num_days = len(df) / 48
        
        metrics = {
            'Method': method,
            'Total Reward (£)': df['total_reward'].sum(),
            'Avg Daily Reward (£)': df['total_reward'].sum() / num_days,
            'Total Revenue (£)': df['revenue'].sum(),
            'Violation Rate (%)': (df['violated'].sum() / len(df)) * 100,
            'Avg SOC (%)': df[['battery1_soc', 'battery2_soc']].mean().mean() * 100,
            'Std Daily Reward': df.groupby(df['datetime'].dt.date)['total_reward'].sum().std()
        }
        comparison.append(metrics)
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Save
    comparison_df.to_csv('results/method_comparison.csv', index=False)
    
    return comparison_df

def visualize_comparison(comparison_df):
    """Create comparison visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = comparison_df['Method'].values
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Total Reward
    axes[0, 0].bar(methods, comparison_df['Total Reward (£)'], color=colors, alpha=0.8)
    axes[0, 0].set_title('Total Reward', fontweight='bold')
    axes[0, 0].set_ylabel('Reward (£)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Avg Daily Reward
    axes[0, 1].bar(methods, comparison_df['Avg Daily Reward (£)'], color=colors, alpha=0.8)
    axes[0, 1].set_title('Average Daily Reward', fontweight='bold')
    axes[0, 1].set_ylabel('Reward (£/day)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Violation Rate
    axes[0, 2].bar(methods, comparison_df['Violation Rate (%)'], color=colors, alpha=0.8)
    axes[0, 2].set_title('Constraint Violation Rate', fontweight='bold')
    axes[0, 2].set_ylabel('Violation Rate (%)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Total Revenue
    axes[1, 0].bar(methods, comparison_df['Total Revenue (£)'], color=colors, alpha=0.8)
    axes[1, 0].set_title('Total Revenue (Energy Arbitrage)', fontweight='bold')
    axes[1, 0].set_ylabel('Revenue (£)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Average SOC
    axes[1, 1].bar(methods, comparison_df['Avg SOC (%)'], color=colors, alpha=0.8)
    axes[1, 1].set_title('Average State of Charge', fontweight='bold')
    axes[1, 1].set_ylabel('SOC (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Reward Stability (Std)
    axes[1, 2].bar(methods, comparison_df['Std Daily Reward'], color=colors, alpha=0.8)
    axes[1, 2].set_title('Reward Stability (Lower is Better)', fontweight='bold')
    axes[1, 2].set_ylabel('Std Dev (£)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Hierarchical vs Baseline Methods Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig('results/figures/method_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison visualization saved")
    plt.show()

if __name__ == "__main__":
    print("="*80)
    print("BASELINE COMPARISON ANALYSIS")
    print("="*80)
    
    # Load results
    all_results = load_all_results()
    
    if len(all_results) > 1:
        # Compare
        comparison_df = compare_metrics(all_results)
        
        # Visualize
        visualize_comparison(comparison_df)
        
        print("\n✓✓✓ Comparison complete!")
        print("Results saved to: results/method_comparison.csv")
    else:
        print("\n⚠ Need at least 2 methods to compare")
        print("Run validation scripts for each baseline first")