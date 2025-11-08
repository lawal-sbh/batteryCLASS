# src/settlement_engine.py
class SettlementEngine:
    def __init__(self):
        self.episode_data = []
    
    def record_episode(self, commander_plan, actual_performance, stability_impact):
        """Record end-of-day settlement data"""
        planned_value = commander_plan.get('expected_value', 0)
        actual_economic = actual_performance.get('economic', 0)
        
        episode = {
            'planned_value': planned_value,
            'actual_value': actual_economic + actual_performance.get('stability', 0),
            'actual_economic': actual_economic,
            'actual_stability': actual_performance.get('stability', 0),
            'stability_impact': stability_impact,
            'flexibility_value': actual_economic - planned_value,
            'battery_utilization': actual_performance.get('utilization', 0),
            'strategy_type': commander_plan.get('strategy_type', 'unknown')
        }
        self.episode_data.append(episode)
        return episode
    
    def calculate_commander_feedback(self):
        """Calculate how Commander should update its strategy"""
        if len(self.episode_data) < 1:
            return {
                "strategy_update": "maintain", 
                "confidence": 0.5,
                "message": "Insufficient data for learning"
            }
        
        recent = self.episode_data[-1]
        flexibility_premium = recent['flexibility_value']
        stability_impact = recent['stability_impact']
        
        # Decision logic based on performance
        if flexibility_premium > 200 and abs(stability_impact) < 1000:
            return {
                "strategy_update": "more_aggressive", 
                "confidence": 0.8,
                "reason": "High flexibility value with low stability cost"
            }
        elif flexibility_premium < -100 or abs(stability_impact) > 3000:
            return {
                "strategy_update": "more_conservative", 
                "confidence": 0.7,
                "reason": "Poor economic performance or high stability cost"
            }
        else:
            return {
                "strategy_update": "maintain", 
                "confidence": 0.6,
                "reason": "Stable performance"
            }
    
    def get_training_summary(self):
        """Get summary of training progress"""
        if len(self.episode_data) == 0:
            return "No training data available"
        
        total_episodes = len(self.episode_data)
        avg_flexibility = sum(ep['flexibility_value'] for ep in self.episode_data) / total_episodes
        avg_stability = sum(ep['actual_stability'] for ep in self.episode_data) / total_episodes
        
        return {
            "total_episodes": total_episodes,
            "average_flexibility_value": avg_flexibility,
            "average_stability_impact": avg_stability,
            "latest_strategy": self.episode_data[-1]['strategy_type'] if total_episodes > 0 else "N/A"
        }