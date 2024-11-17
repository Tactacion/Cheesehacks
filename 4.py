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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import folium
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
import requests
from scipy import stats
import pickle
import base64
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="EVChargePulse Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #0FB881;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.15s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .stProgress > div > div > div {
        background-color: #0FB881;
    }
    .stMetric .label {
        font-size: 14px;
        font-weight: bold;
    }
    .stMetric .value {
        font-size: 24px;
        color: #0FB881;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'preferences': {
            'max_price': 0.30,
            'min_battery': 20,
            'preferred_stations': [],
            'schedule': []
        },
        'achievements': [],
        'points': 0,
        'history': [],
        'community_rank': 'Green Novice'
    }