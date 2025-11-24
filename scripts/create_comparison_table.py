"""
scripts/create_comparison_table.py
Generate comprehensive comparison of all methods
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load all available results"""
    results = {}
    
    # SAC results
    try:
        sac_metrics = json.load(open('results/metrics_sac.json'))
        results['SAC (Single Battery)'] = sac_metrics
        print("✓ Loaded SAC results")
    except:
        print("✗ SAC results not found")
    
    # Rule-based results
    try:
        rule_metrics = json.load(open('results/metrics_rule_based_final.json'))
        results['Rule-Based Optimized'] = rule_metrics
        print("✓ Loaded rule-based results")
    except:
        print("✗ Rule-based results not found")
    
    # Original hierarchical results
    try:
        hier_metrics = json.load(open('results/metrics_hierarchical.json'))
        results['Hierarchical (Original)'] = hier_metrics
        print("✓ Loaded hierarchical results")
    except:
        print("✗ Hierarchical results not found")
    
    return results

def create_comparison_table(results):
    """Create comparison table"""
    
    comparison = []
    
    for method, metrics in results.items():
        row = {
            'Method': method,
            'Avg Daily Reward (£)': metrics.get('avg_reward_per_day', 0),
            'Total Reward (£)': metrics.get('total_reward', 0),
            'Violation Rate (%)': metrics.get('violation_rate', 0) * 100,
            'Num Days': metrics.get('num_days', 0)
        }
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    
    # Sort by daily reward
    df = df.sort_values('Avg Daily Reward (£)', ascending=False)
    
    return df

def visualize_comparison(df):
    """Create visualization"""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    methods = df['Method'].values
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # 1. Daily Reward
    axes[0].barh(methods, df['Avg Daily Reward (£)'], color=colors)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('Average Daily Reward (£)')
    axes[0].set_title('Daily Performance', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(df['Avg Daily Reward (£)']):
        axes[0].text(v, i, f' £{v:.2f}', va='center', fontsize=9)
    
    # 2. Violation Rate
    axes[1].barh(methods, df['Violation Rate (%)'], color=colors)
    axes[1].set_xlabel('Violation Rate (%)')
    axes[1].set_title('Constraint Satisfaction', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(df['Violation Rate (%)']):
        axes[1].text(v, i, f' {v:.2f}%', va='center', fontsize=9)
    
    # 3. Total Reward
    axes[2].barh(methods, df['Total Reward (£)'] / 1000, color=colors)
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    axes[2].set_xlabel('Total Reward (£ thousands)')
    axes[2].set_title('Total Performance', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(df['Total Reward (£)'] / 1000):
        axes[2].text(v, i, f' £{v:.1f}k', va='center', fontsize=9)
    
    plt.suptitle('Battery Dispatch Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def generate_latex_table(df):
    """Generate LaTeX table for thesis"""
    
    latex = r"""\begin{table}[h]
\centering
\caption{Performance Comparison of Battery Dispatch Methods on UK Grid Data (515 days)}
\label{tab:method_comparison}
\begin{tabular}{lrrr}
\toprule
Method & Avg Daily Reward (£) & Violation Rate (\%) & Total Reward (£) \\
\midrule
"""
    
    for _, row in df.iterrows():
        latex += f"{row['Method']} & {row['Avg Daily Reward (£)']:.2f} & {row['Violation Rate (%)']:.2f} & {row['Total Reward (£)']:,.0f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex

def main():
    print("="*70)
    print("METHOD COMPARISON ANALYSIS")
    print("="*70)
    
    # Load results
    results = load_results()
    
    if len(results) < 2:
        print("\n⚠️ Need at least 2 methods to compare!")
        print("Run validation scripts first:")
        print("  - python src/validation/validate_sac_agent.py")
        print("  - python scripts/final_rule_based_validation.py")
        return
    
    # Create comparison table
    df = create_comparison_table(results)
    
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False))
    
    # Save CSV
    Path('results/comparison').mkdir(parents=True, exist_ok=True)
    df.to_csv('results/comparison/method_comparison.csv', index=False)
    print("\n✓ Table saved: results/comparison/method_comparison.csv")
    
    # Create visualization
    fig = visualize_comparison(df)
    fig.savefig('results/comparison/method_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved: results/comparison/method_comparison.png")
    plt.close()
    
    # Generate LaTeX
    latex = generate_latex_table(df)
    with open('results/comparison/table_latex.tex', 'w') as f:
        f.write(latex)
    print("✓ LaTeX saved: results/comparison/table_latex.tex")
    
    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    best = df.iloc[0]
    worst = df.iloc[-1]
    
    print(f"\n🏆 BEST METHOD: {best['Method']}")
    print(f"   Daily reward: £{best['Avg Daily Reward (£)']:.2f}")
    print(f"   Violations: {best['Violation Rate (%)']:.2f}%")
    
    print(f"\n❌ WORST METHOD: {worst['Method']}")
    print(f"   Daily reward: £{worst['Avg Daily Reward (£)']:.2f}")
    print(f"   Violations: {worst['Violation Rate (%)']:.2f}%")
    
    # Improvement
    if len(df) >= 2:
        improvement = ((best['Avg Daily Reward (£)'] - worst['Avg Daily Reward (£)']) / 
                      abs(worst['Avg Daily Reward (£)']) * 100)
        print(f"\n📈 IMPROVEMENT:")
        print(f"   Best vs Worst: {improvement:.1f}% better")
    
    # SAC specific analysis
    if 'SAC (Single Battery)' in results:
        sac_violation = df[df['Method'] == 'SAC (Single Battery)']['Violation Rate (%)'].values[0]
        if sac_violation == 0:
            print(f"\n✅ SAC ACHIEVEMENT: Zero constraint violations!")
            print(f"   This demonstrates perfect constraint satisfaction")
    
    print("\n" + "="*70)
    print("✓✓✓ COMPARISON COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/comparison/method_comparison.csv")
    print("  - results/comparison/method_comparison.png")
    print("  - results/comparison/table_latex.tex")

if __name__ == "__main__":
    main()