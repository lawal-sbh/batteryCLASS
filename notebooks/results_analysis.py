import matplotlib.pyplot as plt

# Your research results
strategies = ['Human Heuristic', 'AI (Constrained)']
revenues = [4693, 7750]
colors = ['red', 'green']

plt.figure(figsize=(10, 6))
bars = plt.bar(strategies, revenues, color=colors, alpha=0.7)
plt.ylabel('Daily Revenue (£)')
plt.title('AI vs Human Trading Performance\n(Constrained by Physics)')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, revenue in zip(bars, revenues):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'£{revenue:,}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('ai_vs_human_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== RESEARCH SUMMARY ===")
print(f"AI Improvement: +£{7750-4693:,} ({((7750/4693)-1)*100:.1f}%)")
print("Key Achievement: AI learned physically plausible arbitrage")
print("Research Ready: This demonstrates RL viability for energy storage optimization")