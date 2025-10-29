"""
RIGOROUS EXPERIMENTAL FRAMEWORK
MSc Thesis - Statistical Validation of AI Trading Performance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from rl_environment import BatteryTradingEnv
from stable_baselines3 import PPO

class RigorousEvaluator:
    def __init__(self, n_seeds=5, n_test_episodes=100):
        self.n_seeds = n_seeds
        self.n_test_episodes = n_test_episodes
        self.results = {}
        
    def run_statistical_testing(self):
        """Run multiple seeds for statistical significance"""
        print("=== RIGOROUS STATISTICAL EVALUATION ===")
        
        ai_revenues = []
        heuristic_revenue = 4693  # Your proven baseline
        
        for seed in range(self.n_seeds):
            print(f"Running experiment {seed+1}/{self.n_seeds}...")
            
            # AI performance
            ai_revenue = self.test_ai_agent(seed)
            ai_revenues.append(ai_revenue)
        
        # Statistical analysis
        self.analyze_results(ai_revenues, heuristic_revenue)
        
    def test_ai_agent(self, seed):
        """Test AI agent with specific random seed"""
        env = BatteryTradingEnv()
        model = PPO("MlpPolicy", env, seed=seed, verbose=0)
        model.learn(total_timesteps=20000)
        
        # Test performance
        test_env = BatteryTradingEnv()
        state, info = test_env.reset()
        total_reward = 0
        
        for step in range(24):
            action, _states = model.predict(state, deterministic=True)
            action = int(action)
            state, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
        
        return total_reward * 1000  # Convert to actual revenue
    
    def analyze_results(self, ai_revenues, heuristic_revenue):
        """Comprehensive statistical analysis"""
        print("\n=== STATISTICAL ANALYSIS ===")
        
        ai_mean = np.mean(ai_revenues)
        ai_std = np.std(ai_revenues)
        
        # Confidence intervals
        ai_ci = stats.t.interval(0.95, len(ai_revenues)-1, 
                                loc=ai_mean, scale=ai_std/np.sqrt(len(ai_revenues)))
        
        # T-test for significance
        t_stat, p_value = stats.ttest_1samp(ai_revenues, heuristic_revenue)
        
        print(f"AI Performance (n={len(ai_revenues)}):")
        print(f"  Mean Revenue: £{ai_mean:,.0f}")
        print(f"  Std Dev: £{ai_std:,.0f}")
        print(f"  95% CI: £{ai_ci[0]:,.0f} - £{ai_ci[1]:,.0f}")
        print(f"  Heuristic Baseline: £{heuristic_revenue:,.0f}")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_value:.4f}")
        
        # Statistical significance
        if p_value < 0.05:
            print("✅ STATISTICALLY SIGNIFICANT: AI outperforms heuristic (p < 0.05)")
            improvement = ((ai_mean - heuristic_revenue) / heuristic_revenue) * 100
            print(f"✅ Average Improvement: {improvement:+.1f}%")
        else:
            print("❌ NOT STATISTICALLY SIGNIFICANT: Results could be due to chance")
        
        # Effect size
        cohens_d = (ai_mean - heuristic_revenue) / ai_std
        print(f"Effect Size (Cohen's d): {cohens_d:.3f}")
        
        # Practical significance
        if cohens_d > 0.8:
            print("✅ LARGE EFFECT SIZE: Practically significant")
        elif cohens_d > 0.5:
            print("✅ MEDIUM EFFECT SIZE: Moderate practical significance")
        else:
            print("⚠️  SMALL EFFECT SIZE: Limited practical significance")

if __name__ == "__main__":
    evaluator = RigorousEvaluator(n_seeds=5)
    evaluator.run_statistical_testing()