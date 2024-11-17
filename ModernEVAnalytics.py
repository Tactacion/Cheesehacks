# Import all required libraries first
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

class ModernEVAnalytics:
    """Enhanced analytics and visualization system for EV charging"""
    def __init__(self):
        plt.style.use('dark_background')
        self.fig_size = (15, 9)
        
        self.colors = {
            'primary': '#3B82F6',    # Blue
            'secondary': '#10B981',   # Green
            'accent': '#F59E0B',      # Amber
            'warning': '#EF4444',     # Red
            'success': '#34D399',     # Emerald
            'purple': '#8B5CF6',      # Purple
            'text': '#F3F4F6',        # Light gray
            'grid': '#374151',        # Dark gray
            'background': '#111827'    # Very dark blue
        }
        
        plt.rcParams.update({
            'figure.figsize': self.fig_size,
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'axes.grid': True,
            'axes.edgecolor': self.colors['grid'],
            'axes.labelcolor': self.colors['text'],
            'grid.color': self.colors['grid'],
            'grid.alpha': 0.3,
            'text.color': self.colors['text'],
            'xtick.color': self.colors['text'],
            'ytick.color': self.colors['text'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10
        })

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
        ax.set_title(title, pad=20, color=self.colors['text'])
        ax.set_xlabel(xlabel, color=self.colors['text'])
        ax.set_ylabel(ylabel, color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.grid(True, color=self.colors['grid'], alpha=0.3)
        
        if ax.get_legend():
            ax.legend(facecolor=self.colors['background'],
                     edgecolor=self.colors['grid'])

    def create_dashboard(self, charging_data: Dict, battery_data: Dict):
        """Create enhanced dashboard of EV charging metrics"""
        fig = plt.figure(figsize=self.fig_size, dpi=200)
        fig.patch.set_facecolor(self.colors['background'])
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('EV Charging Analytics Dashboard', 
                    fontsize=16, 
                    color=self.colors['text'],
                    y=0.95)
        
        self._plot_load_profile(charging_data, fig.add_subplot(gs[0, :]))
        self._plot_battery_health(battery_data, fig.add_subplot(gs[1, 0]))
        self._plot_efficiency(charging_data, fig.add_subplot(gs[1, 1]))
        self._plot_cost_analysis(charging_data, fig.add_subplot(gs[1, 2]))
        self._plot_charging_patterns(charging_data, fig.add_subplot(gs[2, 0]))
        self._plot_power_quality(charging_data, fig.add_subplot(gs[2, 1]))
        self._plot_environmental_impact(charging_data, fig.add_subplot(gs[2, 2]))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def _plot_load_profile(self, data: Dict, ax):
        """Enhanced charging load profile plot"""
        times = data['timestamps']
        loads = data['loads']
        
        line = ax.plot(times, loads, color=self.colors['primary'], 
                      label='Actual Load', alpha=0.7, linewidth=2)[0]
        
        self._add_gradient_fill(ax, line, alpha=0.3)
        
        window = 12
        if len(loads) > window:
            moving_avg = pd.Series(loads).rolling(window=window).mean()
            ax.plot(times, moving_avg, '--', color=self.colors['secondary'],
                   label='Predicted Load', alpha=0.7, linewidth=2)
        
        peak_load = max(loads)
        peak_time = times[loads.index(peak_load)]
        ax.annotate(f'Peak: {peak_load:.1f} kW',
                   xy=(peak_time, peak_load),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5',
                           fc=self.colors['accent'],
                           ec='none',
                           alpha=0.3),
                   arrowprops=dict(arrowstyle='->',
                                 color=self.colors['accent']))
        
        self._style_axis(ax, 'Charging Load Profile', 'Time', 'Load (kW)')

    def _plot_battery_health(self, data: Dict, ax):
        """Enhanced battery health distribution plot"""
        health_values = data['health_values']
        
        n, bins, patches = ax.hist(health_values, bins=20, 
                                 color=self.colors['primary'],
                                 alpha=0.6, edgecolor='none')
        
        fracs = n / n.max()
        norm = plt.Normalize(fracs.min(), fracs.max())
        for frac, patch in zip(fracs, patches):
            color = plt.cm.Blues(norm(frac))
            patch.set_facecolor(color)
        
        mean_health = np.mean(health_values)
        std_health = np.std(health_values)
        
        stats_text = f'Mean: {mean_health:.1f}%\nStd: {std_health:.1f}%'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor=self.colors['background'],
                         edgecolor=self.colors['grid'],
                         alpha=0.8,
                         boxstyle='round,pad=0.5'))
        
        self._style_axis(ax, 'Battery Health Distribution', 'Health (%)', 'Count')

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

    def _plot_cost_analysis(self, data: Dict, ax):
        """Enhanced cost analysis plot"""
        times = data['timestamps']
        costs = data['costs']
        prices = data['prices']
        
        bars = ax.bar(times, costs, alpha=0.6, label='Charging Cost')
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0, max(costs)))
        for bar in bars:
            bar.set_facecolor(sm.to_rgba(bar.get_height()))
        
        ax2 = ax.twinx()
        ax2.plot(times, prices, color=self.colors['warning'],
                label='Energy Price', linewidth=2)
        
        total_cost = sum(costs)
        avg_cost = np.mean(costs)
        stats_text = f'Total: ${total_cost:.2f}\nAvg: ${avg_cost:.2f}/session'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor=self.colors['background'],
                         edgecolor=self.colors['grid'],
                         alpha=0.8,
                         boxstyle='round,pad=0.5'))
        
        self._style_axis(ax, 'Cost Analysis', 'Time', 'Cost ($)')
        ax2.set_ylabel('Price ($/kWh)', color=self.colors['warning'])

    def _plot_charging_patterns(self, data: Dict, ax):
        """Enhanced charging patterns plot"""
        times = data['timestamps']
        loads = data['loads']
        
        hours = [t.hour for t in times]
        hour_avg = pd.DataFrame({'hour': hours, 'load': loads}).groupby('hour').mean()
        
        line = ax.plot(hour_avg.index, hour_avg['load'],
                      color=self.colors['primary'], linewidth=2)[0]
        self._add_gradient_fill(ax, line, alpha=0.4)
        
        peak_hour = hour_avg['load'].idxmax()
        peak_load = hour_avg['load'].max()
        ax.annotate(f'Peak Hour: {peak_hour}:00',
                   xy=(peak_hour, peak_load),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5',
                           fc=self.colors['accent'],
                           ec='none',
                           alpha=0.3),
                   arrowprops=dict(arrowstyle='->',
                                 color=self.colors['accent']))
        
        self._style_axis(ax, 'Daily Charging Pattern', 'Hour of Day', 'Average Load (kW)')

    def _plot_power_quality(self, data: Dict, ax):
        """Enhanced power quality metrics plot"""
        times = data['timestamps']
        voltage = np.array(data['voltage']) / 230.0
        current = np.array(data['current']) / 100.0
        power_factor = data['power_factor']
        
        ln1 = ax.plot(times, voltage, color=self.colors['primary'],
                     label='Voltage (p.u.)', linewidth=2)[0]
        self._add_gradient_fill(ax, ln1, alpha=0.2)
        
        ln2 = ax.plot(times, current, color=self.colors['warning'],
                     label='Current (p.u.)', linewidth=2)[0]
        
        ln3 = ax.plot(times, power_factor, color=self.colors['success'],
                     label='Power Factor', linewidth=2)[0]
        
        ax.axhline(y=0.95, color=self.colors['accent'], linestyle='--',
                  alpha=0.5, label='PF Threshold')
        
        self._style_axis(ax, 'Power Quality Metrics', 'Time', 'Per Unit Value')

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

def main():
    """Main function to demonstrate the enhanced visualization system"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure matplotlib for high-DPI displays
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    
    # Generate sample data
    charging_data, battery_data = generate_sample_data()
    
    # Create analytics dashboard with modern styling
    analyzer = ModernEVAnalytics()
    fig = analyzer.create_dashboard(charging_data, battery_data)
    
    # Save high-resolution image for presentation
    fig.savefig('ev_analytics_dashboard.png', 
                dpi=200, 
                bbox_inches='tight',
                facecolor=analyzer.colors['background'])
    
    # Display the dashboard
    plt.show()

if __name__ == "__main__":
    main()