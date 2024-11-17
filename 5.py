import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from datetime import datetime, timedelta
import json
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import random
from scipy import stats

class EVChargePulse:
    def __init__(self):
        self.charging_stations = self._initialize_stations()
        self.ml_model = RandomForestRegressor()
        self.community_data = self._initialize_community()
        self._train_prediction_model()

    def _initialize_stations(self):
        return {
            f"Station_{i}": {
                "id": f"CS{i:03d}",
                "location": {
                    "lat": 40.7128 + random.uniform(-0.1, 0.1),
                    "lon": -74.0060 + random.uniform(-0.1, 0.1)
                },
                "charger_types": ["Level 2", "DC Fast"],
                "price_per_kwh": round(random.uniform(0.12, 0.25), 2),
                "available": random.randint(1, 4),
                "total_ports": random.randint(4, 8),
                "rating": random.uniform(4.0, 5.0),
                "reviews": [],
                "amenities": ["Wifi", "Restroom", "Shopping"],
                "wait_time": random.randint(0, 30)
            } for i in range(1, 11)
        }

    def _initialize_community(self):
        return {
            "leaderboard": [
                {"user": f"EVUser_{i}", "points": random.randint(1000, 5000),
                 "co2_saved": random.uniform(100, 500),
                 "badges": ["Early Adopter", "Green Warrior"]}
                for i in range(1, 21)
            ],
            "events": [
                {"name": "EV Meetup 2024", "date": "2024-12-01", "location": "Central Park"},
                {"name": "Green Drive Day", "date": "2024-12-15", "location": "Downtown"}
            ],
            "tips": [
                "Charge during off-peak hours to save money",
                "Regular maintenance extends battery life",
                "Use regenerative braking to maximize range"
            ]
        }

    def _train_prediction_model(self):
        """Train prediction model for charging time estimation"""
        X = np.random.rand(1000, 4)  # Features: battery level, temperature, time of day, day of week
        y = np.random.rand(1000) * 60  # Charging time in minutes
        self.ml_model.fit(X, y)

def create_3d_usage_chart(data):
    """Create 3D visualization of charging patterns"""
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    usage = np.random.rand(len(days), len(hours)) * 100

    fig = go.Figure(data=[go.Surface(z=usage)])
    fig.update_layout(
        title='Weekly Charging Patterns',
        scene=dict(
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            zaxis_title='Usage %',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        template="plotly_dark"
    )
    return fig

def main():
    st.set_page_config(
        page_title="EVnergy",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize system
    if 'ev_system' not in st.session_state:
        st.session_state.ev_system = EVChargePulse()

    # Custom CSS
    st.markdown("""
        <style>
        .css-1d391kg {
            background-color: #1E1E1E;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }
        .metric-card {
            background-color: #2C2C2C;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("EVnergy")
        selected_page = st.radio(
            "Navigation",
            ["Smart Charging", "Community", "Analytics", "Impact"]
        )

        # User profile
        st.sidebar.subheader("Your Profile")
        vehicle_model = st.selectbox(
            "Vehicle Model",
            ["Tesla Model 3", "Chevrolet Bolt", "Nissan Leaf", "Ford Mustang Mach-E"]
        )
        battery_level = st.slider("Current Battery Level", 0, 100, 50)
        
        # Gamification elements
        st.sidebar.subheader("Your Achievements")
        st.progress(0.7, "Level Progress")
        st.write("Green Warrior | 🌟 3,240 points")

    # Main content
    if selected_page == "Smart Charging":
        st.title("Smart Charging Assistant")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Range", f"{battery_level * 3.5:.0f} km", "45 km until next charge")
        with col2:
            st.metric("Optimal Charge Time", "22:00", "Saves $5.20")
        with col3:
            st.metric("Network Status", "92% Available", "5 stations nearby")

        # 3D Charging Pattern Visualization
        st.subheader("Charging Patterns Analysis")
        fig_3d = create_3d_usage_chart(st.session_state.ev_system.charging_stations)
        st.plotly_chart(fig_3d, use_container_width=True)

        # Station Map
        st.subheader("Nearby Charging Stations")
        stations_df = pd.DataFrame([
            {
                'latitude': station['location']['lat'],
                'longitude': station['location']['lon'],
                'station': name,
                'available': station['available'],
                'price': station['price_per_kwh']
            }
            for name, station in st.session_state.ev_system.charging_stations.items()
        ])

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v10',
            initial_view_state=pdk.ViewState(
                latitude=40.7128,
                longitude=-74.0060,
                zoom=12,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ColumnLayer',
                    data=stations_df,
                    get_position=['longitude', 'latitude'],
                    get_elevation=['available', 10],
                    elevation_scale=100,
                    radius=100,
                    get_fill_color=['available * 25', 'available * 15', 50, 140],
                    pickable=True,
                    auto_highlight=True
                )
            ]
        ))

    elif selected_page == "Community":
        st.title("EV Community Hub")
        
        # Community Features
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Leaderboard")
            leaderboard_df = pd.DataFrame(st.session_state.ev_system.community_data['leaderboard'])
            st.dataframe(leaderboard_df, hide_index=True)
            
            st.subheader("Community Events")
            for event in st.session_state.ev_system.community_data['events']:
                with st.expander(f"🎉 {event['name']} - {event['date']}"):
                    st.write(f"📍 Location: {event['location']}")
                    st.button("Join Event", key=event['name'])
        
        with col2:
            st.subheader("EV Tips")
            for tip in st.session_state.ev_system.community_data['tips']:
                st.info(tip)
            
            st.subheader("Share Your Experience")
            station_review = st.selectbox("Select Station", list(st.session_state.ev_system.charging_stations.keys()))
            rating = st.slider("Rating", 1, 5, 5)
            review = st.text_area("Your Review")
            if st.button("Submit Review"):
                st.success("Review submitted successfully!")

    elif selected_page == "Analytics":
        st.title("Charging Analytics")
        
        # Create sample charging history
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        charging_history = pd.DataFrame({
            'date': dates,
            'energy': np.random.normal(30, 5, 30),
            'cost': np.random.normal(15, 3, 30),
            'efficiency': np.random.normal(90, 5, 30)
        })
        
        # Interactive charts
        st.subheader("Charging History")
        fig = px.line(charging_history, x='date', y=['energy', 'cost', 'efficiency'],
                     title='Charging Trends')
        st.plotly_chart(fig, use_container_width=True)
        
        # Charging patterns
        st.subheader("Usage Patterns")
        pattern_fig = go.Figure()
        hours = np.arange(24)
        pattern_fig.add_trace(go.Scatter(
            x=hours,
            y=np.random.normal(50, 10, 24),
            fill='tozeroy',
            name='Weekday'
        ))
        pattern_fig.add_trace(go.Scatter(
            x=hours,
            y=np.random.normal(40, 15, 24),
            fill='tozeroy',
            name='Weekend'
        ))
        st.plotly_chart(pattern_fig, use_container_width=True)

    else:  # Impact page
        st.title("Environmental Impact")
        
        # Impact metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CO2 Saved", "1,234 kg", "12% increase")
        with col2:
            st.metric("Trees Equivalent", "45 trees", "3 trees this month")
        with col3:
            st.metric("Green Energy Used", "85%", "5% increase")
        
        # Impact visualization
        impact_data = pd.DataFrame({
            'Month': pd.date_range(start='2024-01-01', periods=12, freq='M'),
            'CO2_Saved': np.cumsum(np.random.normal(100, 10, 12)),
            'Green_Energy': np.random.uniform(70, 90, 12)
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=impact_data['Month'], y=impact_data['CO2_Saved'],
                      name="CO2 Saved", line=dict(color="#4CAF50")),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=impact_data['Month'], y=impact_data['Green_Energy'],
                      name="Green Energy %", line=dict(color="#2196F3")),
            secondary_y=True
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()