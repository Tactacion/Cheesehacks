import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
import json
import time

class EVChargePulse:
    def __init__(self):
        self.charging_stations = {
            "Station A": {"location": "Downtown", "capacity": 10, "available": 7},
            "Station B": {"location": "Shopping Mall", "capacity": 15, "available": 4},
            "Station C": {"location": "Business Park", "capacity": 8, "available": 2}
        }
        
    def predict_wait_time(self, station, current_queue):
        """Simulate wait time prediction using basic queueing theory"""
        avg_charging_time = 45  # minutes
        return max(0, (current_queue * avg_charging_time) // self.charging_stations[station]["available"])
    
    def calculate_optimal_price(self, demand, grid_load, renewable_ratio):
        """Dynamic pricing based on current conditions"""
        base_price = 0.15  # base price per kWh
        demand_factor = 1 + (demand / 100)
        grid_factor = 1 + (grid_load / 100)
        green_discount = 1 - (renewable_ratio / 2)
        return round(base_price * demand_factor * grid_factor * green_discount, 3)
    
    def recommend_charging_time(self, user_schedule, grid_data):
        """Recommend optimal charging slots based on user schedule and grid conditions"""
        slots = []
        for hour in range(24):
            score = 0
            # Higher score for off-peak hours
            if 23 <= hour <= 5:
                score += 3
            elif 10 <= hour <= 16:
                score += 2
            # Higher score for high renewable energy availability
            score += grid_data['renewable_ratio'][hour] * 2
            # Lower score for high grid load
            score -= grid_data['grid_load'][hour] / 100
            slots.append({'hour': hour, 'score': score})
        
        return sorted(slots, key=lambda x: x['score'], reverse=True)[:3]

def main():
    st.set_page_config(page_title="EVChargePulse", layout="wide")
    
    # Initialize session state for user preferences
    if 'preferences' not in st.session_state:
        st.session_state.preferences = {
            'max_price': 0.30,
            'min_battery': 20,
            'preferred_stations': [],
            'schedule': []
        }
    
    # Sidebar for user profile and preferences
    with st.sidebar:
        st.title("🔋 EVChargePulse")
        
        # User Profile
        st.subheader("Driver Profile")
        vehicle_model = st.selectbox("Vehicle Model", 
            ["Tesla Model 3", "Chevrolet Bolt", "Nissan Leaf", "Other"])
        battery_capacity = st.number_input("Battery Capacity (kWh)", 40, 100, 60)
        current_charge = st.slider("Current Charge (%)", 0, 100, 50)
        
        # Smart Charging Preferences
        st.subheader("Smart Charging")
        st.session_state.preferences['max_price'] = st.slider(
            "Max Price per kWh ($)", 0.1, 0.5, 0.3, 0.01)
        st.session_state.preferences['min_battery'] = st.slider(
            "Minimum Battery Level (%)", 10, 50, 20)
    
    # Main content
    st.title("Smart EV Charging Assistant")
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["🎯 Smart Charging", "📊 Analysis", "🌍 Impact"])
    
    # Initialize EVChargePulse
    ev_pulse = EVChargePulse()
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Real-time Station Status")
            for station, details in ev_pulse.charging_stations.items():
                cols = st.columns([3, 2, 2, 2])
                with cols[0]:
                    st.write(f"**{station}** ({details['location']})")
                with cols[1]:
                    st.write(f"Available: {details['available']}/{details['capacity']}")
                with cols[2]:
                    wait_time = ev_pulse.predict_wait_time(station, 
                        details['capacity'] - details['available'])
                    st.write(f"Wait: ~{wait_time} mins")
                with cols[3]:
                    if st.button("Navigate", key=station):
                        st.success(f"Navigation started to {station}")
        
        with col2:
            st.subheader("Smart Charging Plan")
            remaining_range = (current_charge / 100) * (battery_capacity * 4)  # rough estimate
            st.metric("Estimated Range", f"{remaining_range:.0f} km")
            
            if current_charge < st.session_state.preferences['min_battery']:
                st.warning("⚠️ Battery level below minimum threshold!")
                
                # Simulate grid data
                grid_data = {
                    'grid_load': np.random.uniform(30, 90, 24),
                    'renewable_ratio': np.random.uniform(0.2, 0.8, 24)
                }
                
                best_slots = ev_pulse.recommend_charging_time(
                    st.session_state.preferences['schedule'], grid_data)
                
                st.write("Recommended Charging Times:")
                for slot in best_slots:
                    price = ev_pulse.calculate_optimal_price(
                        70, grid_data['grid_load'][slot['hour']], 
                        grid_data['renewable_ratio'][slot['hour']])
                    st.info(
                        f"🕐 {slot['hour']:02d}:00 - ${price}/kWh "
                        f"(Green Energy: {grid_data['renewable_ratio'][slot['hour']]*100:.0f}%)"
                    )
    
    with tab2:
        st.subheader("Charging Analytics")
        
        # Simulate historical charging data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        charging_data = pd.DataFrame({
            'date': dates,
            'energy': np.random.normal(30, 5, 30),
            'cost': np.random.normal(15, 3, 30),
            'green_ratio': np.random.uniform(0.3, 0.8, 30)
        })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Avg. Daily Energy",
                f"{charging_data['energy'].mean():.1f} kWh",
                f"{charging_data['energy'].mean() - 30:.1f} kWh"
            )
        with col2:
            st.metric(
                "Avg. Daily Cost",
                f"${charging_data['cost'].mean():.2f}",
                f"${charging_data['cost'].mean() - 20:.2f}"
            )
        with col3:
            st.metric(
                "Green Energy Usage",
                f"{charging_data['green_ratio'].mean()*100:.1f}%",
                f"{(charging_data['green_ratio'].mean() - 0.5)*100:.1f}%"
            )
    
    with tab3:
        st.subheader("Environmental Impact")
        
        # Calculate environmental metrics
        co2_saved = charging_data['energy'].sum() * charging_data['green_ratio'].mean() * 0.4
        trees_equivalent = co2_saved * 0.1
        
        st.markdown(f"""
        ### Your Impact
        - **CO2 Emissions Avoided**: {co2_saved:.1f} kg
        - **Equivalent to**: {trees_equivalent:.1f} trees planted
        - **Green Energy Ratio**: {charging_data['green_ratio'].mean()*100:.1f}%
        """)
        
        # Gamification elements
        st.progress(min(charging_data['green_ratio'].mean(), 1.0))
        if charging_data['green_ratio'].mean() > 0.7:
            st.success("🌟 You've earned the Green Champion badge!")
        
        # Community impact
        st.write("### Community Impact")
        if st.button("Share Impact"):
            st.success("Your impact has been shared with the EV community! 🌍")
        
        # Future predictions
        st.write("### Future Impact Prediction")
        months = st.slider("Forecast Period (Months)", 1, 12, 6)
        projected_savings = co2_saved * months
        st.info(f"At your current usage pattern, you could save {projected_savings:.1f} kg of CO2 in the next {months} months!")

if __name__ == "__main__":
    main()