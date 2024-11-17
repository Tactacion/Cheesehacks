import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.dates as mdates

class EVAnalytics:
    """Analytics and visualization system for EV charging"""
    def __init__(self):
        # Use default style instead of seaborn
        plt.style.use('default')
        self.fig_size = (20, 12)
        # Set up custom style
        plt.rcParams['figure.figsize'] = self.fig_size
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def create_dashboard(self, charging_data: Dict, battery_data: Dict):
        """Create comprehensive dashboard of EV charging metrics"""
        fig = plt.figure(figsize=self.fig_size)
        
        # Create subplot grid
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Create all subplots
        self._plot_load_profile(charging_data, fig.add_subplot(gs[0, :]))
        self._plot_battery_health(battery_data, fig.add_subplot(gs[1, 0]))
        self._plot_efficiency(charging_data, fig.add_subplot(gs[1, 1]))
        self._plot_cost_analysis(charging_data, fig.add_subplot(gs[1, 2]))
        self._plot_charging_patterns(charging_data, fig.add_subplot(gs[2, 0]))
        self._plot_power_quality(charging_data, fig.add_subplot(gs[2, 1]))
        self._plot_environmental_impact(charging_data, fig.add_subplot(gs[2, 2]))
        
        plt.tight_layout()
        plt.show()
        
    def _plot_load_profile(self, data: Dict, ax):
        """Plot charging load profile with predictions"""
        times = data['timestamps']
        loads = data['loads']
        
        # Plot actual load
        ax.plot(times, loads, 'b-', label='Actual Load', alpha=0.7)
        
        # Add moving average prediction
        window = 12
        if len(loads) > window:
            moving_avg = pd.Series(loads).rolling(window=window).mean()
            ax.plot(times, moving_avg, 'r--', label='Predicted Load', alpha=0.7)
        
        # Add peak annotations
        peak_load = max(loads)
        peak_time = times[loads.index(peak_load)]
        ax.annotate(f'Peak: {peak_load:.1f} kW',
                   xy=(peak_time, peak_load),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           fc='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->'))
        
        ax.set_title('Charging Load Profile')
        ax.set_xlabel('Time')
        ax.set_ylabel('Load (kW)')
        ax.grid(True)
        ax.legend()
        
    def _plot_battery_health(self, data: Dict, ax):
        """Plot battery health distribution"""
        health_values = data['health_values']
        
        # Create histogram
        ax.hist(health_values, bins=20, color='blue', alpha=0.5)
        
        # Add statistical annotations
        mean_health = np.mean(health_values)
        std_health = np.std(health_values)
        
        stats_text = f'Mean: {mean_health:.1f}%\nStd: {std_health:.1f}%'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title('Battery Health Distribution')
        ax.set_xlabel('Health (%)')
        ax.set_ylabel('Count')
        ax.grid(True)
        
    def _plot_efficiency(self, data: Dict, ax):
        """Plot charging efficiency metrics"""
        times = data['timestamps']
        efficiency = data['efficiency']
        temperature = data['temperature']
        
        # Create twin axes for efficiency and temperature
        ax2 = ax.twinx()
        
        # Plot efficiency
        ln1 = ax.plot(times, efficiency, 'b-', label='Efficiency')
        ln2 = ax2.plot(times, temperature, 'r-', label='Temperature')
        
        ax.set_title('Charging Efficiency')
        ax.set_xlabel('Time')
        ax.set_ylabel('Efficiency (%)')
        ax2.set_ylabel('Temperature (°C)')
        
        # Add combined legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right')
        ax.grid(True)
        
    def _plot_cost_analysis(self, data: Dict, ax):
        """Plot charging cost analysis"""
        times = data['timestamps']
        costs = data['costs']
        prices = data['prices']
        
        # Plot costs as bars
        ax.bar(times, costs, alpha=0.5, label='Charging Cost')
        
        # Add price line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(times, prices, 'r-', label='Energy Price')
        
        # Add cost statistics
        total_cost = sum(costs)
        avg_cost = np.mean(costs)
        stats_text = f'Total: ${total_cost:.2f}\nAvg: ${avg_cost:.2f}/session'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title('Charging Cost Analysis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cost ($)')
        ax2.set_ylabel('Price ($/kWh)')
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True)

    def _plot_charging_patterns(self, data: Dict, ax):
        """Plot charging patterns as line plot"""
        times = data['timestamps']
        loads = data['loads']
        
        # Create hour of day averages
        hours = [t.hour for t in times]
        hour_avg = pd.DataFrame({'hour': hours, 'load': loads}).groupby('hour').mean()
        
        ax.plot(hour_avg.index, hour_avg['load'], 'b-', linewidth=2)
        ax.fill_between(hour_avg.index, hour_avg['load'], alpha=0.3)
        
        ax.set_title('Average Daily Charging Pattern')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Load (kW)')
        ax.grid(True)
        
    def _plot_power_quality(self, data: Dict, ax):
        """Plot power quality metrics"""
        times = data['timestamps']
        voltage = np.array(data['voltage']) / 230.0  # Normalize to 1.0
        current = np.array(data['current']) / 100.0  # Normalize to 1.0
        power_factor = data['power_factor']
        
        # Plot metrics
        ax.plot(times, voltage, 'b-', label='Voltage (p.u.)', alpha=0.7)
        ax.plot(times, current, 'r-', label='Current (p.u.)', alpha=0.7)
        ax.plot(times, power_factor, 'g-', label='Power Factor', alpha=0.7)
        
        ax.set_title('Power Quality Metrics')
        ax.set_xlabel('Time')
        ax.set_ylabel('Per Unit Value')
        ax.legend()
        ax.grid(True)
        
    def _plot_environmental_impact(self, data: Dict, ax):
        """Plot environmental impact metrics"""
        times = data['timestamps']
        emissions = data['emissions']
        renewable_ratio = data['renewable_ratio']
        
        # Create twin axes
        ax2 = ax.twinx()
        
        # Plot emissions and renewable ratio
        ln1 = ax.plot(times, emissions, 'r-', label='CO2 Emissions')
        ln2 = ax2.plot(times, renewable_ratio, 'g-', label='Renewable Ratio')
        
        ax.set_title('Environmental Impact')
        ax.set_xlabel('Time')
        ax.set_ylabel('CO2 Emissions (kg)')
        ax2.set_ylabel('Renewable Ratio')
        
        # Add combined legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right')
        ax.grid(True)

def generate_sample_data(num_days: int = 7) -> Tuple[Dict, Dict]:
    """Generate sample data for visualization"""
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=num_days)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_days * 24)]
    
    # Generate charging data
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
    
    # Generate battery data
    battery_data = {
        'health_values': np.random.normal(90, 5, 100)  # 100 batteries
    }
    
    return charging_data, battery_data

def main():
    # Generate sample data
    charging_data, battery_data = generate_sample_data()
    
    # Create analytics dashboard
    analyzer = EVAnalytics()
    analyzer.create_dashboard(charging_data, battery_data)

if __name__ == "__main__":
    main()