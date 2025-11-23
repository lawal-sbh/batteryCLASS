"""
src/validation/validate_hierarchical.py
Validates hierarchical Commander-Tactician agent on real UK grid data

Usage:
    python src/validation/validate_hierarchical.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# ============================================
# AGENT ARCHITECTURE DEFINITIONS
# ============================================

class CommanderNetwork(nn.Sequential):
    """
    Commander network - Day-ahead strategic planning
    Architecture: 5D input → 256 hidden → 256 hidden → 48D output
    Inherits directly from Sequential to match training format
    """
    def __init__(self, state_dim=5, action_dim=48, hidden_dim=256):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

class TacticianNetwork(nn.Sequential):
    """
    Tactician network - Real-time execution
    Architecture: 5D input → 128 hidden → 128 hidden → 2D output
    Inherits directly from Sequential to match training format
    """
    def __init__(self, state_dim=5, action_dim=2, hidden_dim=128):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

class HierarchicalBatteryAgent:
    """
    Commander-Tactician Hierarchical Architecture
    - Commander: Plans entire day (48 settlement periods)
    - Tactician: Executes immediate actions (2 batteries)
    """
    
    def __init__(self, commander_path, tactician_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load Commander checkpoint
        print(f"Loading Commander from: {commander_path}")
        commander_checkpoint = torch.load(commander_path, map_location=self.device, weights_only=False)
        
        # Extract policy state dict from full checkpoint
        if isinstance(commander_checkpoint, dict) and 'policy_state_dict' in commander_checkpoint:
            print("  ✓ Detected full training checkpoint")
            commander_state = commander_checkpoint['policy_state_dict']
            print(f"  ✓ Trained for {commander_checkpoint.get('episode', 'unknown')} episodes")
            print(f"  ✓ State size: {commander_checkpoint.get('state_size', 'unknown')}")
            print(f"  ✓ Action size: {commander_checkpoint.get('action_size', 'unknown')}")
        else:
            commander_state = commander_checkpoint
        
        # Initialize and load Commander
        self.commander = CommanderNetwork().to(self.device)
        self.commander.load_state_dict(commander_state)
        self.commander.eval()
        print(f"  ✓ Commander loaded successfully")
        
        # Load Tactician checkpoint
        print(f"\nLoading Tactician from: {tactician_path}")
        tactician_checkpoint = torch.load(tactician_path, map_location=self.device, weights_only=False)
        
        # Extract policy state dict from full checkpoint
        if isinstance(tactician_checkpoint, dict) and 'policy_state_dict' in tactician_checkpoint:
            print("  ✓ Detected full training checkpoint")
            tactician_state = tactician_checkpoint['policy_state_dict']
            print(f"  ✓ Trained for {tactician_checkpoint.get('episode', 'unknown')} episodes")
            print(f"  ✓ State size: {tactician_checkpoint.get('state_size', 'unknown')}")
            print(f"  ✓ Action size: {tactician_checkpoint.get('action_size', 'unknown')}")
        else:
            tactician_state = tactician_checkpoint
        
        # Initialize and load Tactician
        self.tactician = TacticianNetwork().to(self.device)
        self.tactician.load_state_dict(tactician_state)
        self.tactician.eval()
        print(f"  ✓ Tactician loaded successfully")
        
        # Hierarchical coordination state
        self.commander_target = None
        self.steps_since_command = 0
        self.command_interval = 12  # Commander updates every 6 hours (12 periods)
        
        print(f"\n✓✓✓ Hierarchical Agent Ready!")
        print(f"  - Commander update interval: {self.command_interval} periods (6 hours)")
        print(f"  - Device: {self.device}")
        
    def predict(self, state_dict, deterministic=True):
        """
        Hierarchical prediction with Commander-Tactician coordination
        
        Commander: Plans entire day (48 actions)
        Tactician: Executes current period (2 actions for 2 batteries)
        
        Args:
            state_dict: Dict with keys: 'hour', 'price', 'soc1', 'soc2', 'grid_stress'
        Returns:
            actions: numpy array [battery1_action, battery2_action] in [-1, 1]
        """
        with torch.no_grad():
            # Create state vector (5D)
            state = torch.FloatTensor([
                state_dict['hour'] / 24,
                state_dict['price'] / 100,
                state_dict['soc1'],
                state_dict['soc2'],
                state_dict['grid_stress']
            ]).unsqueeze(0).to(self.device)
            
            # Commander generates day-ahead plan (48 actions) periodically
            # This simulates day-ahead planning every 6 hours
            if self.steps_since_command % self.command_interval == 0:
                self.commander_target = self.commander(state).cpu().numpy()[0]
            
            # Tactician executes immediate actions (2 actions for 2 batteries)
            action = self.tactician(state).cpu().numpy()[0]
            
            self.steps_since_command += 1
            
            return action  # [battery1_action, battery2_action] in [-1, 1]

# ============================================
# DUAL BATTERY ENVIRONMENT
# ============================================

class DualBatteryEnvironment:
    """
    Environment for 2-battery system with UK grid constraints
    Multi-objective optimization: arbitrage + degradation + grid support
    """
    def __init__(self, 
                 capacity_mwh_per_battery=5,
                 power_limit_mw_per_battery=1,
                 efficiency=0.95,
                 degradation_cost=0.01):
        self.capacity = capacity_mwh_per_battery
        self.power_limit = power_limit_mw_per_battery
        self.efficiency = efficiency
        self.degradation_cost = degradation_cost
        
        # State for 2 batteries
        self.soc = [0.5, 0.5]
        self.total_energy_cycled = [0, 0]
        
    def reset(self):
        """Reset environment to initial state"""
        self.soc = [0.5, 0.5]
        self.total_energy_cycled = [0, 0]
        return self.soc.copy()
        
    def step(self, actions, price, grid_stress=0.5):
        """
        Execute one step in the environment
        
        Args:
            actions: [action1, action2] in [-1, 1] for each battery
            price: Current electricity price (£/MWh)
            grid_stress: Grid stress indicator [0, 1]
            
        Returns:
            new_socs: Updated state of charge for both batteries
            total_reward: Combined reward from both batteries
            violated: Whether any constraints were violated
            info: Additional information dict
        """
        rewards = []
        new_socs = []
        violations = []
        
        for i, action in enumerate(actions):
            # Scale action to power
            power = action * self.power_limit  # MW
            energy = power * 0.5  # MWh (half-hour settlement)
            
            # Apply efficiency
            if energy > 0:  # Charging
                energy_actual = energy * self.efficiency
            else:  # Discharging
                energy_actual = energy / self.efficiency
            
            # Update SOC
            new_soc = self.soc[i] + (energy_actual / self.capacity)
            new_soc = np.clip(new_soc, 0.0, 1.0)
            
            # Check violations
            violated = (new_soc <= 0.01 or new_soc >= 0.99)
            
            # Multi-objective reward:
            # 1. Energy arbitrage revenue (buy low, sell high)
            revenue = -energy * price
            
            # 2. Degradation cost (proportional to throughput)
            degradation = abs(energy) * self.degradation_cost
            
            # 3. Grid support bonus (discharge during high stress)
            grid_support = 0
            if grid_stress > 0.7 and energy < 0:  # Discharging during stress
                grid_support = abs(energy) * 10  # Bonus for grid support
            
            # Combined reward
            reward = revenue - degradation + grid_support
            
            # Penalty for violations
            if violated:
                reward -= 100
            
            # Update state
            self.soc[i] = new_soc
            self.total_energy_cycled[i] += abs(energy)
            
            rewards.append(reward)
            new_socs.append(new_soc)
            violations.append(violated)
        
        return new_socs, sum(rewards), any(violations), {
            'battery1_soc': new_socs[0],
            'battery2_soc': new_socs[1],
            'battery1_reward': rewards[0],
            'battery2_reward': rewards[1],
            'total_cycled': sum(self.total_energy_cycled)
        }

# ============================================
# VALIDATION FUNCTIONS
# ============================================

def load_validation_data(data_path):
    """Load and prepare UK grid data for validation"""
    print("Loading validation data...")
    data = pd.read_csv(data_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Calculate grid stress indicator (normalized demand)
    data['grid_stress'] = (data['TSD'] - data['TSD'].min()) / (data['TSD'].max() - data['TSD'].min())
    
    return data

def create_episodes(data, start_date='2024-06-01'):
    """
    Split data into daily episodes for validation
    
    Args:
        data: Full dataset
        start_date: Start date for test period
        
    Returns:
        episodes: List of daily dataframes (48 periods each)
    """
    test_data = data[data['datetime'] >= start_date].copy()
    
    episodes = []
    for date in test_data['datetime'].dt.date.unique():
        episode = test_data[test_data['datetime'].dt.date == date].copy()
        if len(episode) == 48:  # Complete day only
            episodes.append(episode.reset_index(drop=True))
    
    return episodes

def run_validation(agent, episodes, output_dir='results'):
    """
    Run hierarchical agent validation on episodes
    
    Args:
        agent: HierarchicalBatteryAgent instance
        episodes: List of daily episodes
        output_dir: Directory to save results
        
    Returns:
        results_df: DataFrame with all results
        metrics: Dictionary of summary metrics
    """
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize environment
    env = DualBatteryEnvironment(
        capacity_mwh_per_battery=5,
        power_limit_mw_per_battery=1
    )
    
    # Storage for results
    results = {
        'datetime': [],
        'demand': [],
        'price': [],
        'grid_stress': [],
        'battery1_action': [],
        'battery2_action': [],
        'battery1_power': [],
        'battery2_power': [],
        'battery1_soc': [],
        'battery2_soc': [],
        'total_reward': [],
        'revenue': [],
        'violated': [],
        'commander_update': []
    }
    
    print(f"\n{'='*70}")
    print(f"RUNNING HIERARCHICAL VALIDATION")
    print(f"{'='*70}")
    print(f"Episodes: {len(episodes)}")
    print(f"Architecture: Commander-Tactician")
    print(f"Batteries: 2 x 5 MWh / 1 MW")
    print(f"{'='*70}\n")
    
    total_reward = 0
    total_violations = 0
    
    for ep_idx, episode in enumerate(episodes):
        socs = env.reset()
        
        for t in range(len(episode)):
            row = episode.iloc[t]
            
            # Create state for agent
            state = {
                'hour': row['hour'],
                'price': row['system_price'],
                'soc1': socs[0],
                'soc2': socs[1],
                'grid_stress': row['grid_stress']
            }
            
            # Get hierarchical action
            commander_updated = (agent.steps_since_command % agent.command_interval == 0)
            actions = agent.predict(state)
            
            # Step environment
            socs, reward, violated, info = env.step(
                actions, 
                row['system_price'],
                row['grid_stress']
            )
            
            # Store results
            results['datetime'].append(row['datetime'])
            results['demand'].append(row['TSD'])
            results['price'].append(row['system_price'])
            results['grid_stress'].append(row['grid_stress'])
            results['battery1_action'].append(actions[0])
            results['battery2_action'].append(actions[1])
            results['battery1_power'].append(actions[0] * env.power_limit)
            results['battery2_power'].append(actions[1] * env.power_limit)
            results['battery1_soc'].append(socs[0])
            results['battery2_soc'].append(socs[1])
            results['total_reward'].append(reward)
            results['revenue'].append(-sum(actions) * env.power_limit * 0.5 * row['system_price'])
            results['violated'].append(violated)
            results['commander_update'].append(commander_updated)
            
            total_reward += reward
            if violated:
                total_violations += 1
        
        if (ep_idx + 1) % 30 == 0:
            print(f"  Completed {ep_idx + 1}/{len(episodes)} episodes...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    num_days = len(episodes)
    total_revenue = results_df['revenue'].sum()
    
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Test period:           {num_days} days")
    print(f"   Total reward:          £{total_reward:,.2f}")
    print(f"   Total revenue:         £{total_revenue:,.2f}")
    print(f"   Avg reward/day:        £{total_reward/num_days:.2f}")
    print(f"   Constraint violations:  {total_violations} ({total_violations/(num_days*48)*100:.2f}%)")
    
    print(f"\n⚡ BATTERY 1 STATS:")
    print(f"   Avg SOC:               {results_df['battery1_soc'].mean()*100:.1f}%")
    print(f"   Min SOC:               {results_df['battery1_soc'].min()*100:.1f}%")
    print(f"   Max SOC:               {results_df['battery1_soc'].max()*100:.1f}%")
    
    print(f"\n⚡ BATTERY 2 STATS:")
    print(f"   Avg SOC:               {results_df['battery2_soc'].mean()*100:.1f}%")
    print(f"   Min SOC:               {results_df['battery2_soc'].min()*100:.1f}%")
    print(f"   Max SOC:               {results_df['battery2_soc'].max()*100:.1f}%")
    
    print(f"\n🎯 HIERARCHICAL COORDINATION:")
    print(f"   Commander updates:     {results_df['commander_update'].sum()}")
    print(f"   Update frequency:      Every {agent.command_interval} periods (6 hours)")
    
    # Save results
    output_file = f"{output_dir}/validation_hierarchical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved: {output_file}")
    
    # Save metrics
    metrics = {
        'total_reward': float(total_reward),
        'avg_reward_per_day': float(total_reward / num_days),
        'total_revenue': float(total_revenue),
        'violation_rate': float(total_violations / (num_days * 48)),
        'num_days': num_days,
        'num_violations': int(total_violations)
    }
    
    metrics_file = f"{output_dir}/metrics_hierarchical.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return results_df, metrics

def visualize_results(results_df, output_dir='results'):
    """Create comprehensive visualizations of validation results"""
    print("\nCreating visualizations...")
    
    # Create figures directory
    fig_dir = Path(output_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
    
    # Sample week for detailed view
    week_data = results_df.iloc[:48*7]
    
    # 1. Price and Combined Power
    ax1 = fig.add_subplot(gs[0, :])
    ax1_twin = ax1.twinx()
    ax1.plot(week_data['datetime'], week_data['price'], 'b-', label='Price', linewidth=2)
    total_power = week_data['battery1_power'] + week_data['battery2_power']
    ax1_twin.bar(week_data['datetime'], total_power, alpha=0.3, color='green', label='Total Power')
    ax1.set_ylabel('Price (£/MWh)', color='b', fontsize=10)
    ax1_twin.set_ylabel('Combined Power (MW)', color='g', fontsize=10)
    ax1.set_title('Hierarchical Dispatch: Price and Combined Actions (Sample Week)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 2. Individual Battery Actions
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(week_data['datetime'], week_data['battery1_power'], label='Battery 1', linewidth=2)
    ax2.plot(week_data['datetime'], week_data['battery2_power'], label='Battery 2', 
             linewidth=2, alpha=0.7)
    commander_updates = week_data[week_data['commander_update']]
    ax2.scatter(commander_updates['datetime'], [0]*len(commander_updates), 
                color='red', s=100, marker='^', label='Commander Update', zorder=5)
    ax2.set_ylabel('Power (MW)', fontsize=10)
    ax2.set_title('Individual Battery Actions (Commander Updates Marked)', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 3. SOC Trajectories
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(week_data['datetime'], week_data['battery1_soc']*100, 
             'purple', linewidth=2, label='Battery 1')
    ax3.plot(week_data['datetime'], week_data['battery2_soc']*100, 
             'orange', linewidth=2, label='Battery 2')
    ax3.axhline(y=10, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=90, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_ylabel('SOC (%)', fontsize=10)
    ax3.set_title('State of Charge Trajectories', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Grid Stress Response
    ax4 = fig.add_subplot(gs[2, 1])
    ax4_twin = ax4.twinx()
    ax4.plot(week_data['datetime'], week_data['grid_stress'], 'r-', 
             label='Grid Stress', linewidth=2)
    ax4_twin.plot(week_data['datetime'], total_power, 'g-', 
                  label='Response', linewidth=2, alpha=0.7)
    ax4.set_ylabel('Grid Stress', color='r', fontsize=10)
    ax4_twin.set_ylabel('Power Response (MW)', color='g', fontsize=10)
    ax4.set_title('Grid Stress and Agent Response', fontsize=11, fontweight='bold')
    ax4.axhline(y=0.7, color='r', linestyle='--', alpha=0.3)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative Reward
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(results_df['datetime'], results_df['total_reward'].cumsum(), 
             'darkgreen', linewidth=2)
    ax5.set_ylabel('Cumulative Reward (£)', fontsize=10)
    ax5.set_title('Cumulative Reward Over Time', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Daily Rewards
    ax6 = fig.add_subplot(gs[3, 1])
    daily_rewards = results_df.groupby(results_df['datetime'].dt.date)['total_reward'].sum()
    ax6.bar(range(len(daily_rewards)), daily_rewards.values, alpha=0.7, color='teal')
    ax6.set_ylabel('Daily Reward (£)', fontsize=10)
    ax6.set_xlabel('Day', fontsize=10)
    ax6.set_title('Daily Reward Distribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. SOC Distribution
    ax7 = fig.add_subplot(gs[4, 0])
    ax7.hist([results_df['battery1_soc']*100, results_df['battery2_soc']*100], 
             bins=30, label=['Battery 1', 'Battery 2'], alpha=0.7)
    ax7.set_xlabel('SOC (%)', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('SOC Distribution', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Coordination Effectiveness
    ax8 = fig.add_subplot(gs[4, 1])
    results_df['periods_since_update'] = 0
    count = 0
    for i in range(len(results_df)):
        if results_df.iloc[i]['commander_update']:
            count = 0
        results_df.iloc[i, results_df.columns.get_loc('periods_since_update')] = count
        count += 1
    
    update_performance = results_df.groupby('periods_since_update')['total_reward'].mean()
    if len(update_performance) > 12:
        update_performance = update_performance.iloc[:12]
    ax8.plot(update_performance.index, update_performance.values, 'o-', linewidth=2)
    ax8.set_xlabel('Periods Since Commander Update', fontsize=10)
    ax8.set_ylabel('Avg Reward (£)', fontsize=10)
    ax8.set_title('Coordination Effectiveness', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Hierarchical Multi-Objective RL - Validation Results', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    viz_file = fig_dir / 'validation_visualization.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {viz_file}")
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'data/uk_battery_dispatch_complete_data.csv'
    COMMANDER_PATH = 'models/commander/best_model.pth'
    TACTICIAN_PATH = 'models/tactician/best_model.pth'
    OUTPUT_DIR = 'results'
    
    print("="*70)
    print("HIERARCHICAL BATTERY DISPATCH VALIDATION")
    print("Commander-Tactician Architecture")
    print("="*70)
    
    # Load agent
    print("\nLoading hierarchical agent...")
    agent = HierarchicalBatteryAgent(
        commander_path=COMMANDER_PATH,
        tactician_path=TACTICIAN_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Load data
    data = load_validation_data(DATA_PATH)
    print(f"✓ Loaded data: {len(data):,} rows")
    print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    
    # Create episodes
    episodes = create_episodes(data, start_date='2024-06-01')
    print(f"✓ Created {len(episodes)} validation episodes")
    
    # Run validation
    results_df, metrics = run_validation(agent, episodes, OUTPUT_DIR)
    
    # Visualize
    visualize_results(results_df, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print("✓✓✓ VALIDATION COMPLETE ✓✓✓")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Review results in: {OUTPUT_DIR}")
    print(f"  2. Implement baseline comparisons")
    print(f"  3. Run statistical analysis")
    print(f"  4. Start writing paper results section")