# notebooks/analyze_training.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_training_progress():
    """Analyze the training results and identify improvements needed"""
    print("=== TRAINING RESULTS ANALYSIS ===")
    
    # Load training progress
    progress_file = "models/training_progress_episode_1000.json"
    
    if not os.path.exists(progress_file):
        print("❌ No training progress file found")
        return
    
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    rewards = progress['rewards']
    economic = progress['economic_rewards']
    stability = progress['stability_rewards']
    
    print(f"📊 Training Episodes: {len(rewards)}")
    print(f"💰 Final Economic Reward: £{economic[-1]:.1f}")
    print(f"⚡ Final Stability Reward: £{stability[-1]:.1f}")
    print(f"🎯 Final Total Reward: £{rewards[-1]:.1f}")
    
    # Analyze learning trends
    first_100_avg = np.mean(rewards[:100])
    last_100_avg = np.mean(rewards[-100:])
    improvement = last_100_avg - first_100_avg
    
    print(f"📈 Learning Improvement: £{improvement:.1f}")
    print(f"   First 100 episodes: £{first_100_avg:.1f}")
    print(f"   Last 100 episodes: £{last_100_avg:.1f}")
    
    # Identify issues
    print("\n🔍 IDENTIFIED ISSUES:")
    
    if economic[-1] > 2000 and stability[-1] < -5000:
        print("❌ Economic-stability imbalance: Too focused on profit")
        print("   Solution: Increase stability weight in rewards")
    
    if last_100_avg > 0:
        print("✅ Positive learning: Agents found profitable strategies")
    else:
        print("❌ Negative rewards: Need reward function tuning")
    
    # Create improvement plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Reward progression
    plt.subplot(2, 2, 1)
    plt.plot(rewards, alpha=0.7)
    plt.title('Total Reward Progression')
    plt.xlabel('Episode')
    plt.ylabel('Reward (£)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Economic vs Stability
    plt.subplot(2, 2, 2)
    episodes = list(range(len(economic)))
    plt.plot(episodes, economic, label='Economic', alpha=0.7)
    plt.plot(episodes, stability, label='Stability', alpha=0.7)
    plt.title('Economic vs Stability Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward (£)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Moving average
    plt.subplot(2, 2, 3)
    window = 50
    moving_avg = [np.mean(rewards[i:i+window]) for i in range(len(rewards)-window)]
    plt.plot(moving_avg)
    plt.title(f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (£)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Improvement analysis
    plt.subplot(2, 2, 4)
    quarter = len(rewards) // 4
    quarters = [1, 2, 3, 4]
    quarter_avgs = [
        np.mean(rewards[:quarter]),
        np.mean(rewards[quarter:2*quarter]),
        np.mean(rewards[2*quarter:3*quarter]),
        np.mean(rewards[3*quarter:])
    ]
    plt.bar(quarters, quarter_avgs)
    plt.title('Average Reward by Training Quarter')
    plt.xlabel('Training Quarter')
    plt.ylabel('Average Reward (£)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Analysis plot saved: models/training_analysis.png")
    
    return progress

def generate_improvement_recommendations():
    """Generate specific recommendations for improvement"""
    print("\n🎯 RECOMMENDATIONS FOR NEXT ITERATION:")
    
    recommendations = [
        "1. REWARD FUNCTION: Increase stability weight from 0.01 to 0.02",
        "2. ACTION PENALTY: Add penalty for extreme SOC values (0% or 100%)",
        "3. EXPLORATION: Reduce exploration noise as training progresses",
        "4. STABILITY BONUS: Add positive rewards for good grid behavior",
        "5. SOC TARGETS: Commander should enforce minimum SOC reserves"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n🔧 QUICK FIXES TO IMPLEMENT:")
    print("   - Update stability_weight in environment to 0.02")
    print("   - Add SOC penalty: -10 * (SOC < 0.1 or SOC > 0.9)")
    print("   - Reduce exploration noise from 0.1 to 0.05 after 500 episodes")

if __name__ == "__main__":
    progress = analyze_training_progress()
    generate_improvement_recommendations()