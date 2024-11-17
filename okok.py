import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.dates as mdates
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
    
    return charging_data, {}

class ModernEVAnalytics:
    def __init__(self):
        plt.style.use('dark_background')
        self.fig_size = (20, 12)
        
        self.colors = {
            'primary': '#4287f5',     # Bright blue
            'secondary': '#34D399',    # Green
            'accent': '#F59E0B',       # Amber
            'warning': '#EF4444',      # Red
            'success': '#10B981',      # Emerald
            'text': '#FFFFFF',         # Pure white
            'grid': '#2D3748',         # Dark grid
            'background': '#1A202C'    # Dark background
        }
        
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
        """Create enhanced dashboard with simplified layout"""
        fig = plt.figure(figsize=self.fig_size)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create grid layout with 2 rows
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
        
        # Add dashboard title
        fig.suptitle('EV Charging Analytics Dashboard', 
                    fontsize=20, 
                    color=self.colors['text'],
                    y=0.95,
                    weight='bold')
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, :])    # Load profile (full width)
        ax2 = fig.add_subplot(gs[1, 0])    # Efficiency
        ax3 = fig.add_subplot(gs[1, 1])    # Environmental impact
        
        # Plot components
        self._plot_load_profile(charging_data, ax1)
        self._plot_efficiency(charging_data, ax2)
        self._plot_environmental_impact(charging_data, ax3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        return fig

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
        """Enhanced load profile plot"""
        times = data['timestamps']
        loads = data['loads']
        
        line = ax.plot(times, loads, color=self.colors['primary'], 
                      label='Actual Load', alpha=0.8, linewidth=2)[0]
        
        self._add_gradient_fill(ax, line, alpha=0.2)
        
        window = 12
        if len(loads) > window:
            moving_avg = pd.Series(loads).rolling(window=window).mean()
            ax.plot(times, moving_avg, '--', color=self.colors['secondary'],
                   label='Predicted Load', alpha=0.9, linewidth=2)
        
        peak_load = max(loads)
        peak_time = times[loads.index(peak_load)]
        ax.annotate(f'Peak: {peak_load:.1f} kW',
                   xy=(peak_time, peak_load),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5',
                           fc=self.colors['accent'],
                           ec='none',
                           alpha=0.6),
                   arrowprops=dict(arrowstyle='->',
                                 color=self.colors['accent']),
                   fontsize=10,
                   weight='bold')
        
        self._style_axis(ax, 'Charging Load Profile', 'Time', 'Load (kW)')
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

def main():
    """Main function to demonstrate the enhanced visualization system"""
    np.random.seed(42)
    
    # Generate sample data
    charging_data, _ = generate_sample_data()
    
    # Create analytics dashboard
    analyzer = ModernEVAnalytics()
    fig = analyzer.create_dashboard(charging_data)
    
    # Save high-resolution image
    fig.savefig('ev_analytics_dashboard.png', 
                dpi=300,
                bbox_inches='tight',
                facecolor=analyzer.colors['background'])
    
    plt.show()

if __name__ == "__main__":
    main()