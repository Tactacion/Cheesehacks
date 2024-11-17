import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def calculate_charging_cost(charging_time, charging_rate, electricity_rate):
    """Calculate the cost of charging"""
    energy_consumed = charging_time * charging_rate
    return energy_consumed * electricity_rate

def calculate_battery_degradation(initial_capacity, cycles, degradation_rate=0.04):
    """Calculate battery capacity after given number of cycles"""
    return initial_capacity * (1 - degradation_rate * cycles/1000)

def main():
    st.set_page_config(page_title="EV Analysis Dashboard", layout="wide")
    
    # Header
    st.title("Electric Vehicle Analysis Dashboard")
    st.write("Analyze EV performance, costs, and environmental impact")
    
    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    # Vehicle Parameters
    st.sidebar.subheader("Vehicle Specifications")
    battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", 40.0, 100.0, 60.0)
    initial_range = st.sidebar.number_input("Initial Range (km)", 200.0, 600.0, 400.0)
    charging_rate = st.sidebar.number_input("Charging Rate (kW)", 3.0, 350.0, 11.0)
    
    # Usage Parameters
    st.sidebar.subheader("Usage Pattern")
    daily_distance = st.sidebar.number_input("Average Daily Distance (km)", 0.0, 500.0, 50.0)
    electricity_rate = st.sidebar.number_input("Electricity Rate ($/kWh)", 0.0, 1.0, 0.15)
    
    # Calculate key metrics
    energy_per_km = battery_capacity / initial_range
    daily_energy_needed = daily_distance * energy_per_km
    charging_time_needed = daily_energy_needed / charging_rate
    daily_charging_cost = calculate_charging_cost(charging_time_needed, charging_rate, electricity_rate)
    
    # Display Results in Multiple Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Daily Energy Consumption", f"{daily_energy_needed:.2f} kWh")
        
    with col2:
        st.metric("Charging Time Needed", f"{charging_time_needed:.2f} hours")
        
    with col3:
        st.metric("Daily Charging Cost", f"${daily_charging_cost:.2f}")
    
    # Battery Degradation Analysis
    st.header("Battery Degradation Analysis")
    years = range(1, 11)
    cycles_per_year = 365 * daily_distance / initial_range
    capacities = [calculate_battery_degradation(battery_capacity, cycles_per_year * year) for year in years]
    ranges = [cap / energy_per_km for cap in capacities]
    
    # Create degradation plot
    fig_degradation = go.Figure()
    fig_degradation.add_trace(go.Scatter(x=list(years), y=ranges, 
                                       mode='lines+markers',
                                       name='Projected Range'))
    fig_degradation.update_layout(
        title='Projected Range Degradation Over Time',
        xaxis_title='Years',
        yaxis_title='Range (km)',
        showlegend=True
    )
    st.plotly_chart(fig_degradation)
    
    # Cost Analysis
    st.header("Cost Analysis")
    
    # Calculate yearly costs
    yearly_cost = daily_charging_cost * 365
    years = list(range(1, 6))
    cumulative_costs = [yearly_cost * year for year in years]
    
    # Create cost analysis plot
    fig_costs = go.Figure()
    fig_costs.add_trace(go.Bar(x=years, y=cumulative_costs,
                              name='Cumulative Charging Cost'))
    fig_costs.update_layout(
        title='Cumulative Charging Costs Over Years',
        xaxis_title='Years',
        yaxis_title='Cost ($)',
        showlegend=True
    )
    st.plotly_chart(fig_costs)
    
    # Environmental Impact
    st.header("Environmental Impact")
    co2_per_kwh = st.number_input("Grid CO2 Intensity (kg CO2/kWh)", 0.0, 1.0, 0.4)
    yearly_co2 = daily_energy_needed * 365 * co2_per_kwh
    st.metric("Annual CO2 Emissions", f"{yearly_co2:.2f} kg CO2")
    
    # Save configuration
    if st.button("Download Analysis Report"):
        report_data = {
            "Battery Capacity (kWh)": battery_capacity,
            "Initial Range (km)": initial_range,
            "Daily Distance (km)": daily_distance,
            "Daily Energy Consumption (kWh)": daily_energy_needed,
            "Daily Charging Cost ($)": daily_charging_cost,
            "Annual CO2 Emissions (kg)": yearly_co2
        }
        df = pd.DataFrame([report_data])
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="ev_analysis_report.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()