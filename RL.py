import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Neural Networks for DDPG
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        return self.max_action * torch.tanh(self.layer_3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = torch.relu(self.layer_1(state_action))
        q = torch.relu(self.layer_2(q))
        return self.layer_3(q)

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.storage = deque(maxlen=max_size)
        
    def add(self, transition):
        self.storage.append(transition)
        
    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards).reshape(-1, 1), 
                np.array(next_states))

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.memory = ReplayBuffer()
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).data.numpy().flatten()
    
    def train(self, batch_size=64):
        if len(self.memory.storage) < batch_size:
            return
        
        states, actions, rewards, next_states = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # Critic loss
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions)
        target_Q = rewards + 0.99 * target_Q
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
        
        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)
            
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)

class OptimizedEVCharging:
    def __init__(self, state_dim=4, action_dim=1):
        self.ddpg = DDPG(state_dim=state_dim, action_dim=action_dim, max_action=1.0)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def get_state(self, time, load, efficiency, renewable_ratio):
        hour = time.hour / 24.0  # Normalize hour to [0,1]
        load_normalized = load / 100.0  # Normalize load
        return np.array([hour, load_normalized, efficiency/100.0, renewable_ratio])
        
    def optimize_charging(self, state):
        action = self.ddpg.select_action(state)
        return action
        
    def update(self, state, action, reward, next_state):
        self.ddpg.memory.add((state, action, reward, next_state))
        self.ddpg.train()

class ModernEVAnalytics:
    def __init__(self):
        # Previous initialization code remains the same
        plt.style.use('dark_background')
        self.fig_size = (20, 12)
        
        self.colors = {
            'primary': '#4287f5',
            'secondary': '#34D399',
            'accent': '#F59E0B',
            'warning': '#EF4444',
            'success': '#10B981',
            'text': '#FFFFFF',
            'grid': '#2D3748',
            'background': '#1A202C'
        }
        
        # Initialize RL optimizer
        self.optimizer = OptimizedEVCharging()
        
        # Setup matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': self.fig_size,
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'axes.grid': True,
            'axes.edgecolor': self.colors['grid'],
            'axes.labelcolor': self.colors['text'],
            'grid.color': self.colors['grid'],
            'grid.alpha': 0.2,
            'text.color': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 200,
            'savefig.dpi': 200
        })

    def create_dashboard(self, charging_data: Dict):
        """Create enhanced dashboard with RL optimization"""
        # Optimize charging data
        charging_data = self.optimize_and_plot(charging_data)
        
        # Create visualization
        fig = plt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor(self.colors['background'])
        
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
        
        fig.suptitle('AI-Optimized EV Charging Analytics', 
                    fontsize=20, 
                    color=self.colors['text'],
                    y=0.95,
                    weight='bold')
        
        # Create subplots with optimized data
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        self._plot_load_profile(charging_data, ax1)
        self._plot_efficiency(charging_data, ax2)
        self._plot_environmental_impact(charging_data, ax3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        return fig

    def optimize_and_plot(self, data: Dict):
        """Optimize charging and update data with RL recommendations"""
        optimized_loads = []
        states = []
        actions = []
        
        for i in range(len(data['timestamps'])):
            state = self.optimizer.get_state(
                data['timestamps'][i],
                data['loads'][i],
                data['efficiency'][i],
                data['renewable_ratio'][i]
            )
            states.append(state)
            
            # Get optimal action
            action = self.optimizer.optimize_charging(state)
            actions.append(action)
            
            # Calculate optimized load
            optimized_load = data['loads'][i] * (1 + action[0])
            optimized_loads.append(optimized_load)
            
            # Calculate reward based on efficiency and renewable usage
            if i > 0:
                reward = (data['efficiency'][i] / 100.0 * 
                         data['renewable_ratio'][i] -
                         abs(optimized_load - data['loads'][i]) / 100.0)
                self.optimizer.update(states[-2], actions[-2], reward, states[-1])
        
        data['optimized_loads'] = optimized_loads
        return data

    def _add_gradient_fill(self, ax, line, alpha=0.3):
        """Add gradient fill under a line plot"""
        xdata, ydata = line.get_data()
        colormap = plt.cm.Blues
        z = np.array(ydata)
        z = z - z.min()
        z = z / z.max()
        z = np.reshape(z, (1, len(z)))
        
        extent = [xdata[0], xdata[-1], 0, ydata.max()]
        ax.imshow(z, extent=extent, aspect='auto', alpha=alpha,
                cmap=colormap, origin='lower')

    def _style_axis(self, ax, title, xlabel, ylabel):
        """Apply consistent modern styling to axis"""
        ax.set_title(title, pad=20, color=self.colors['text'], fontsize=14, weight='bold')
        ax.set_xlabel(xlabel, color=self.colors['text'], fontsize=12)
        ax.set_ylabel(ylabel, color=self.colors['text'], fontsize=12)
        ax.tick_params(colors=self.colors['text'], labelsize=10)
        ax.grid(True, color=self.colors['grid'], alpha=0.2, linestyle='--')
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
            spine.set_linewidth(0.5)
        
        if ax.get_legend():
            ax.legend(facecolor=self.colors['background'],
                    edgecolor=self.colors['grid'],
                    fontsize=10,
                    loc='upper right')

    def _format_dates(self, ax, rotation=45):
        """Format date labels"""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')

    def _plot_load_profile(self, data: Dict, ax):
        """Enhanced load profile plot with RL optimization"""
        times = data['timestamps']
        loads = data['loads']
        optimized_loads = data['optimized_loads']
        
        # Plot actual load
        line1 = ax.plot(times, loads, color=self.colors['primary'], 
                       label='Actual Load', alpha=0.8, linewidth=2)[0]
        self._add_gradient_fill(ax, line1, alpha=0.2)
        
        # Plot optimized load
        line2 = ax.plot(times, optimized_loads, '--', color=self.colors['success'],
                       label='AI-Optimized Load', alpha=0.9, linewidth=2)[0]
        
        # Calculate and show optimization metrics
        load_reduction = ((np.mean(loads) - np.mean(optimized_loads)) / np.mean(loads)) * 100
        efficiency_gain = load_reduction * np.mean(data['renewable_ratio'])
        
        stats_text = f'Load Reduction: {load_reduction:.1f}%\nEfficiency Gain: {efficiency_gain:.1f}%'
        ax.text(0.02, 0.95, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor=self.colors['background'],
                         edgecolor=self.colors['grid'],
                         alpha=0.8,
                         boxstyle='round,pad=0.5'))
        
        self._style_axis(ax, 'AI-Optimized Charging Load Profile', 'Time', 'Load (kW)')
        self._format_dates(ax)

    def _plot_efficiency(self, data: Dict, ax):
        """Enhanced efficiency metrics plot"""
        times = data['timestamps']
        efficiency = data['efficiency']
        temperature = data['temperature']
        
        ax2 = ax.twinx()
        
        ln1 = ax.plot(times, efficiency, color=self.colors['primary'],
                    label='Efficiency', linewidth=2)[0]
        self._add_gradient_fill(ax, ln1, alpha=0.2)
        
        ln2 = ax2.plot(times, temperature, color=self.colors['warning'],
                    label='Temperature', linewidth=2)[0]
        
        self._style_axis(ax, 'Charging Efficiency', 'Time', 'Efficiency (%)')
        ax2.set_ylabel('Temperature (°C)', color=self.colors['warning'])
        
        lns = [ln1, ln2]
        labs = ['Efficiency', 'Temperature']
        ax.legend(lns, labs, loc='upper right',
                facecolor=self.colors['background'],
                edgecolor=self.colors['grid'])
        self._format_dates(ax)

    def _plot_environmental_impact(self, data: Dict, ax):
        """Enhanced environmental impact plot"""
        times = data['timestamps']
        emissions = data['emissions']
        renewable_ratio = data['renewable_ratio']
        
        ax2 = ax.twinx()
        
        ln1 = ax.plot(times, emissions, color=self.colors['warning'],
                    label='CO2 Emissions', linewidth=2)[0]
        self._add_gradient_fill(ax, ln1, alpha=0.2)
        
        ln2 = ax2.plot(times, renewable_ratio, color=self.colors['success'],
                    label='Renewable Ratio', linewidth=2)[0]
        
        ax2.axhline(y=0.5, color=self.colors['accent'], linestyle='--',
                alpha=0.5, label='Target Ratio')
        
        self._style_axis(ax, 'Environmental Impact', 'Time', 'CO2 Emissions (kg)')
        ax2.set_ylabel('Renewable Ratio', color=self.colors['success'])
        
        lns = [ln1, ln2]
        labs = ['CO2 Emissions', 'Renewable Ratio']
        ax.legend(lns, labs, loc='upper right',
                facecolor=self.colors['background'],
                edgecolor=self.colors['grid'])
        self._format_dates(ax)
        
    
    
    # [Previous plotting methods remain the same]

def generate_sample_data(num_days: int = 7) -> Tuple[Dict, Dict]:
    """Generate sample data for visualization"""
    start_time = datetime.now() - timedelta(days=num_days)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_days * 24)]
    
    charging_data = {
        'timestamps': timestamps,
        'loads': [50 + 30 * np.sin(i/24 * np.pi) + np.random.normal(0, 5) 
                 for i in range(len(timestamps))],
        'efficiency': [92 + np.random.normal(0, 2) for _ in timestamps],
        'temperature': [25 + 10 * np.sin(i/24 * np.pi) + np.random.normal(0, 2)
                       for i in range(len(timestamps))],
        'voltage': [230 + np.random.normal(0, 2) for _ in timestamps],
        'current': [100 + np.random.normal(0, 5) for _ in timestamps],
        'power_factor': [0.95 + np.random.normal(0, 0.02) for _ in timestamps],
        'costs': [10 + np.random.normal(0, 1) for _ in timestamps],
        'prices': [0.15 + 0.05 * np.sin(i/24 * np.pi) for i in range(len(timestamps))],
        'emissions': [5 + np.random.normal(0, 0.5) for _ in timestamps],
        'renewable_ratio': [0.3 + 0.2 * np.sin(i/24 * np.pi) for i in range(len(timestamps))]
    }
    
    battery_data = {
        'health_values': np.random.normal(90, 5, 100)  # 100 batteries
    }
    
    return charging_data, battery_data
class WeatherAwareOptimizer:
    def __init__(self):
        self.weather_impact = {
            'clear': 1.0,
            'cloudy': 0.8,
            'rainy': 0.6,
            'snow': 0.4
        }
        
    def predict_renewable_availability(self, weather_data, solar_capacity):
        return np.array([
            self.weather_impact[condition] * solar_capacity 
            for condition in weather_data
        ])
        
    def optimize_charging_schedule(self, weather_forecast, load_profile):
        # Adjust charging based on predicted renewable energy availability
        pass
class V2GOptimizer:
    def __init__(self):
        self.grid_demand_threshold = 0.8
        self.min_battery_level = 0.2
        
    def calculate_v2g_potential(self, battery_levels, grid_demand):
        """Calculate when EVs can supply power back to grid"""
        v2g_windows = []
        for time, (battery, demand) in enumerate(zip(battery_levels, grid_demand)):
            if battery > self.min_battery_level and demand > self.grid_demand_threshold:
                v2g_windows.append((time, battery - self.min_battery_level))
        return v2g_windows
class PriceOptimizer:
    def __init__(self):
        self.lstm_model = self._build_lstm_model()
        
    def predict_price_trends(self, historical_prices, horizon=24):
        """Predict future electricity prices"""
        return self.lstm_model.predict(historical_prices)
    
    def optimize_charging_cost(self, price_predictions, required_charge):
        """Find optimal charging windows based on predicted prices"""
        pass
class GridBalancer:
    def __init__(self):
        self.grid_capacity = 1000  # kW
        self.safety_margin = 0.1
        
    def distribute_load(self, charging_requests, grid_status):
        """Intelligently distribute charging load across grid"""
        total_capacity = self.grid_capacity * (1 - self.safety_margin)
        prioritized_requests = self._prioritize_requests(charging_requests)
        return self._allocate_power(prioritized_requests, total_capacity)
class BatteryHealthOptimizer:

    def __init__(self):
        self.degradation_model = self._build_degradation_model()
        
    def predict_degradation(self, charging_pattern, temperature, cycles):
        """Predict battery degradation based on usage patterns"""
        features = np.column_stack([charging_pattern, temperature, cycles])
        return self.degradation_model.predict(features)
    
    def optimize_charging_profile(self, current_health, desired_lifespan):
        """Generate optimal charging profile to maximize battery life"""
        pass
class FleetOptimizer:
    def __init__(self):
        self.genetic_algorithm = GeneticOptimizer()
        
    def optimize_fleet_charging(self, vehicles, constraints):
        """Optimize charging schedule for multiple vehicles"""
        return self.genetic_algorithm.solve(vehicles, constraints)
        
class GeneticOptimizer:
    def __init__(self):
        self.population_size = 100
        self.generations = 50
        
    def solve(self, vehicles, constraints):
        """Use genetic algorithm to find optimal charging schedule"""
        pass
class AdvancedDashboard:
    def create_3d_visualization(self, data):
        """Create 3D visualization of charging patterns"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data['time'], data['load'], data['temperature'],
                           c=data['efficiency'], cmap='viridis')
        return fig
        
    def create_heatmap_calendar(self, charging_data):
        """Create calendar heatmap of charging patterns"""
        pass
class RealTimeOptimizer:
    def __init__(self):
        self.current_state = {}
        self.optimization_interval = 300  # 5 minutes
        
    async def optimize_in_realtime(self, charging_session):
        """Continuously optimize charging parameters"""
        while charging_session.is_active:
            current_data = await self.get_telemetry()
            optimal_params = self.calculate_optimal_parameters(current_data)
            await self.apply_parameters(optimal_params)
            
class UserBehaviorLearner:
    def __init__(self):
        self.user_profiles = {}
        
    def learn_user_patterns(self, user_id, charging_history):
        """Learn and adapt to user charging patterns"""
        regular_times = self._identify_regular_charging_times(charging_history)
        typical_duration = self._calculate_typical_duration(charging_history)
        return self._create_user_profile(regular_times, typical_duration)
class EnhancedEVAnalytics(ModernEVAnalytics):
    def __init__(self):
        super().__init__()
        self.weather_optimizer = WeatherAwareOptimizer()
        self.v2g_optimizer = V2GOptimizer()
        self.price_optimizer = PriceOptimizer()
        self.fleet_optimizer = FleetOptimizer()
        self.realtime_optimizer = RealTimeOptimizer()
        self.user_learner = UserBehaviorLearner()


def main():
    """Main function with RL optimization"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate sample data
    charging_data, _ = generate_sample_data()
    
    # Create analytics dashboard with RL optimization
    analyzer = ModernEVAnalytics()
    fig = analyzer.create_dashboard(charging_data)
    
    # Save high-resolution image
    fig.savefig('ev_analytics_dashboard_optimized.png', 
                dpi=300,
                bbox_inches='tight',
                facecolor=analyzer.colors['background'])
    
    plt.show()

if __name__ == "__main__":
    main()