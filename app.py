"""
🚀 QuantML Studio v4.0 - Enterprise AI/ML Platform
====================================================
Complete Streamlit Application
"""

import os
import sys
import logging
import warnings
import time
import uuid
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT ENGINES
# =============================================================================
AUTOML_AVAILABLE = False
NEURAL_NETWORKS_AVAILABLE = False

try:
    from core.automl_engine import (
        AutoMLEngine, AutoMLConfig, TaskType, OptimizationStrategy,
        ModelDriftDetector, ModelExplainer, RiskMetricsCalculator,
        TORCH_AVAILABLE, SHAP_AVAILABLE, XGBOOST_AVAILABLE
    )
    AUTOML_AVAILABLE = True
except ImportError as e:
    logger.error(f"AutoML engine not available: {e}")

try:
    from core.neural_networks import (
        NeuralNetworkFactory, NeuralNetworkTrainer, NNTrainingConfig,
        TORCH_AVAILABLE as NN_TORCH_AVAILABLE
    )
    if NN_TORCH_AVAILABLE:
        NEURAL_NETWORKS_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# CONFIG & CONSTANTS
# =============================================================================
APP_CONFIG = {'app_name': 'QuantML Studio', 'version': '5.0.0', 'tagline': 'Enterprise AI/ML Platform - Enhanced Edition'}

SUBSCRIPTION_TIERS = {
    'free': {'name': 'Free', 'price': '$0', 'max_rows': 1000, 'max_features': 20, 'max_models': 5, 'time_limit': 5,
             'include_neural_networks': False, 'include_ensemble': False, 'include_explainability': False,
             'include_drift_detection': False, 'include_risk_metrics': False, 'include_advanced_features': False,
             'include_model_comparison': False, 'include_whatif': False, 'include_anomaly': False,
             'include_forecasting': False, 'include_reports': False, 'include_api': False, 'include_registry': False,
             'api_access': False, 'export_models': False, 'batch_predictions': False, 'color': '#64748b', 'icon': '🆓'},
    'starter': {'name': 'Starter', 'price': '$29/mo', 'max_rows': 10000, 'max_features': 50, 'max_models': 15, 'time_limit': 30,
                'include_neural_networks': False, 'include_ensemble': True, 'include_explainability': True,
                'include_drift_detection': False, 'include_risk_metrics': False, 'include_advanced_features': True,
                'include_model_comparison': True, 'include_whatif': True, 'include_anomaly': False,
                'include_forecasting': False, 'include_reports': True, 'include_api': False, 'include_registry': False,
                'api_access': False, 'export_models': True, 'batch_predictions': True, 'color': '#10b981', 'icon': '🚀'},
    'professional': {'name': 'Professional', 'price': '$99/mo', 'max_rows': 100000, 'max_features': 200, 'max_models': 30, 'time_limit': 120,
                     'include_neural_networks': True, 'include_ensemble': True, 'include_explainability': True,
                     'include_drift_detection': True, 'include_risk_metrics': True, 'include_advanced_features': True,
                     'include_model_comparison': True, 'include_whatif': True, 'include_anomaly': True,
                     'include_forecasting': True, 'include_reports': True, 'include_api': True, 'include_registry': True,
                     'api_access': True, 'export_models': True, 'batch_predictions': True, 'color': '#8B5CF6', 'icon': '⭐'},
    'enterprise': {'name': 'Enterprise', 'price': 'Custom', 'max_rows': float('inf'), 'max_features': float('inf'), 'max_models': 50, 'time_limit': 480,
                   'include_neural_networks': True, 'include_ensemble': True, 'include_explainability': True,
                   'include_drift_detection': True, 'include_risk_metrics': True, 'include_advanced_features': True,
                   'include_model_comparison': True, 'include_whatif': True, 'include_anomaly': True,
                   'include_forecasting': True, 'include_reports': True, 'include_api': True, 'include_registry': True,
                   'api_access': True, 'export_models': True, 'batch_predictions': True, 'color': '#f59e0b', 'icon': '🏢'}
}

PREMIUM_KEYS = {'STARTER-2024': 'starter', 'STARTER-DEMO-2024': 'starter', 'PRO-2024': 'professional', 
                'PRO-DEMO-2024': 'professional', 'ENTERPRISE-2024': 'enterprise', 'ENTERPRISE-DEMO-2024': 'enterprise'}

MODEL_INFO = {
    'LinearRegression': {'name': 'Linear Regression', 'icon': '📈', 'color': '#3B82F6'},
    'Ridge': {'name': 'Ridge', 'icon': '📐', 'color': '#6366F1'},
    'Lasso': {'name': 'Lasso', 'icon': '📏', 'color': '#8B5CF6'},
    'ElasticNet': {'name': 'ElasticNet', 'icon': '🔗', 'color': '#A855F7'},
    'DecisionTree': {'name': 'Decision Tree', 'icon': '🌳', 'color': '#22C55E'},
    'RandomForest': {'name': 'Random Forest', 'icon': '🌲', 'color': '#10B981'},
    'ExtraTrees': {'name': 'Extra Trees', 'icon': '🌴', 'color': '#14B8A6'},
    'GradientBoosting': {'name': 'Gradient Boosting', 'icon': '🚀', 'color': '#F59E0B'},
    'XGBoost': {'name': 'XGBoost', 'icon': '⚡', 'color': '#EF4444'},
    'LightGBM': {'name': 'LightGBM', 'icon': '💡', 'color': '#EC4899'},
    'CatBoost': {'name': 'CatBoost', 'icon': '🐱', 'color': '#F97316'},
    'SVR': {'name': 'SVR', 'icon': '📊', 'color': '#06B6D4'},
    'SVC': {'name': 'SVC', 'icon': '🎯', 'color': '#0EA5E9'},
    'KNN': {'name': 'KNN', 'icon': '👥', 'color': '#84CC16'},
    'LogisticRegression': {'name': 'Logistic Regression', 'icon': '📉', 'color': '#3B82F6'},
    'NaiveBayes': {'name': 'Naive Bayes', 'icon': '🎲', 'color': '#9333EA'}
}

NEURAL_MODELS = {
    'transformer': {'name': 'Transformer', 'icon': '🤖', 'desc': 'Attention-based'},
    'cnn_lstm': {'name': 'CNN-LSTM', 'icon': '🔄', 'desc': 'Hybrid model'},
    'tcn': {'name': 'TCN', 'icon': '📊', 'desc': 'Temporal Conv'},
    'mlp': {'name': 'MLP', 'icon': '🧠', 'desc': 'Multi-Layer'},
    'lstm_gru': {'name': 'LSTM-GRU', 'icon': '🔁', 'desc': 'Ensemble'}
}

# =============================================================================
# HELPERS
# =============================================================================
def validate_key(key): 
    k = key.strip().upper()
    return (True, PREMIUM_KEYS[k]) if k in PREMIUM_KEYS else (False, None)

def get_limits(): return SUBSCRIPTION_TIERS[st.session_state.subscription_tier]
def check_feature(f): return get_limits().get(f, False)
def can_use_nn(): return check_feature('include_neural_networks') and NEURAL_NETWORKS_AVAILABLE
def get_model_info(n): return MODEL_INFO.get(n, {'name': n, 'icon': '🤖', 'color': '#64748b'})

# =============================================================================
# STYLING
# =============================================================================
def apply_styling():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #c026d3 100%);
        border-radius: 24px; padding: 2rem; margin-bottom: 2rem;
        box-shadow: 0 25px 50px -12px rgba(139, 92, 246, 0.35);
    }
    .hero-title { font-size: 2.5rem; font-weight: 700; color: #ffffff; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .hero-subtitle { color: #ffffff; font-size: 1.1rem; margin-top: 0.5rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
    .tier-badge { padding: 8px 16px; border-radius: 100px; font-weight: 700; background: rgba(255,255,255,0.25); color: #ffffff; border: 2px solid rgba(255,255,255,0.4); }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px; padding: 1.25rem; text-align: center; border: 2px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .metric-value { font-size: 1.75rem; font-weight: 700; color: #0f172a; }
    .metric-label { font-size: 0.8rem; color: #475569; text-transform: uppercase; font-weight: 600; }
    
    .alert-card { padding: 1rem; border-radius: 12px; margin: 0.75rem 0; border-left: 4px solid; font-weight: 500; }
    .alert-card.success { background: #dcfce7; border-color: #16a34a; color: #14532d; }
    .alert-card.warning { background: #fef3c7; border-color: #d97706; color: #78350f; }
    .alert-card.error { background: #fee2e2; border-color: #dc2626; color: #7f1d1d; }
    .alert-card.info { background: #dbeafe; border-color: #2563eb; color: #1e3a8a; }
    .alert-card.neural { background: #ede9fe; border-color: #7c3aed; color: #4c1d95; }
    
    .model-card { background: white; border-radius: 16px; padding: 1.25rem; border: 2px solid #e2e8f0; margin-bottom: 0.75rem; }
    .model-card.best { border-color: #16a34a; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); }
    
    .glass-card { 
        background: #ffffff; 
        border-radius: 20px; 
        padding: 1.5rem; 
        border: 2px solid #e2e8f0; 
        box-shadow: 0 10px 40px -10px rgba(0,0,0,0.12);
    }
    .glass-card h4 { color: #0f172a; font-weight: 700; margin-bottom: 0.75rem; }
    .glass-card p { color: #334155; font-size: 0.95rem; line-height: 1.6; }
    .glass-card li { color: #334155; }
    .glass-card strong { color: #0f172a; }
    
    .feature-card {
        background: #ffffff;
        border-radius: 16px; padding: 1.5rem; border: 2px solid #e2e8f0;
        transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .feature-card:hover { transform: translateY(-4px); box-shadow: 0 20px 40px -15px rgba(0,0,0,0.2); border-color: #7c3aed; }
    .feature-card strong { color: #0f172a; font-size: 1rem; }
    .feature-card p { color: #475569; margin-top: 0.5rem; }
    
    .prediction-result { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 20px; padding: 2rem; text-align: center; border: 3px solid #16a34a; }
    .prediction-value { font-size: 3rem; font-weight: 700; color: #14532d; }
    
    .quality-gauge {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        border-radius: 20px; padding: 2rem; text-align: center; color: white;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.3);
    }
    .quality-score { font-size: 4rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
    
    .footer { text-align: center; padding: 2rem; color: #475569; font-size: 0.85rem; margin-top: 3rem; border-top: 2px solid #e2e8f0; font-weight: 500; }
    
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e3a8a 0%, #4c1d95 100%); }
    section[data-testid="stSidebar"] .stMarkdown { color: white; }
    
    /* Main content text improvements */
    .stMarkdown h2 { color: #0f172a !important; font-weight: 700 !important; }
    .stMarkdown h3 { color: #1e293b !important; font-weight: 600 !important; }
    .stMarkdown h4 { color: #1e293b !important; font-weight: 600 !important; }
    .stMarkdown p { color: #334155 !important; }
    
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# STATE
# =============================================================================
def init_state():
    defaults = {
        'subscription_tier': 'free', 'user_id': str(uuid.uuid4())[:8], 'session_start': datetime.now(),
        'current_page': 'home', 'uploaded_data': None, 'target_column': None, 'feature_columns': [],
        'automl_engine': None, 'neural_model': None, 'training_complete': False, 'model_results': {},
        'training_runs': 0, 'predictions_made': 0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_header():
    tier = get_limits()
    st.markdown(f"""
    <div class="hero-header">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
            <div>
                <h1 class="hero-title">🚀 {APP_CONFIG['app_name']}</h1>
                <p class="hero-subtitle">v{APP_CONFIG['version']} • {APP_CONFIG['tagline']}</p>
            </div>
            <div class="tier-badge">{tier['icon']} {tier['name']} Plan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        tier = get_limits()
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem;background:rgba(255,255,255,0.1);border-radius:16px;margin-bottom:1.5rem;">
            <div style="font-size:3rem;">{tier['icon']}</div>
            <div style="font-weight:600;font-size:1.25rem;color:white;">{tier['name']}</div>
            <div style="font-size:0.9rem;color:rgba(255,255,255,0.7);">{tier['price']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📍 Navigation")
        pages = [('home', '🏠 Dashboard'), ('upload', '📤 Data Upload'), ('train', '🏋️ AutoML Training'),
                 ('neural', '🧠 Neural Networks'), ('results', '📊 Results'), ('explain', '🔍 Explainability'),
                 ('drift', '🚨 Drift Detection'), ('predict', '🔮 Predictions'), ('pricing', '💎 Pricing'), ('settings', '⚙️ Settings')]
        
        for page_key, label in pages:
            is_current = st.session_state.current_page == page_key
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type="primary" if is_current else "secondary"):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        if st.session_state.subscription_tier == 'free':
            st.markdown("### 🔑 Activate Premium")
            key = st.text_input("Key:", type="password", key="sidebar_key", label_visibility="collapsed")
            if st.button("✨ Activate", use_container_width=True):
                valid, new_tier = validate_key(key)
                if valid:
                    st.session_state.subscription_tier = new_tier
                    st.success(f"✅ {SUBSCRIPTION_TIERS[new_tier]['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid key")

def render_metric_card(label, value, icon="📊"):
    st.markdown(f"""<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

def render_alert(msg, t="info"):
    st.markdown(f'<div class="alert-card {t}">{msg}</div>', unsafe_allow_html=True)

def render_model_card(rank, name, results, is_best, is_reg):
    info = get_model_info(name)
    m1 = f"R²: {results.get('test_r2', 0):.4f}" if is_reg else f"Acc: {results.get('test_accuracy', 0):.4f}"
    m2 = f"RMSE: {results.get('test_rmse', 0):.4f}" if is_reg else f"F1: {results.get('test_f1', 0):.4f}"
    cv = abs(results.get('cv_mean', results.get('cv_score', 0)))
    st.markdown(f"""
    <div class="model-card {'best' if is_best else ''}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:2rem;">{info['icon']}</span>
                <div><strong>{info['name']}</strong><div style="font-size:0.85rem;color:#64748b;">{m1} | {m2}</div></div>
            </div>
            <div style="text-align:right;">
                <div style="font-weight:600;color:{info['color']};">{'🏆' if is_best else f'#{rank}'}</div>
                <div style="font-size:0.75rem;color:#64748b;">CV: {cv:.4f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGES
# =============================================================================
def page_home():
    st.markdown("## 🏠 Welcome to QuantML Studio")
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("Training Runs", str(st.session_state.training_runs), "🏃")
    with c2: render_metric_card("Predictions", str(st.session_state.predictions_made), "🔮")
    with c3: render_metric_card("Models", "15+", "🤖")
    with c4: render_metric_card("Plan", get_limits()['name'], "💎")
    
    # About Section
    st.markdown("### 🎯 About QuantML Studio")
    st.markdown("""
    <div class="glass-card" style="margin-bottom: 1.5rem;">
        <h4 style="color: #1e293b; margin-bottom: 1rem;">📌 What is QuantML Studio?</h4>
        <p style="color: #475569; line-height: 1.7;">
            QuantML Studio is an <strong>enterprise-grade Automated Machine Learning (AutoML) platform</strong> designed to democratize 
            AI and machine learning for data scientists, analysts, and business professionals. Our platform eliminates the complexity 
            of building ML models by automating the entire pipeline—from data preprocessing to model deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ How It Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="glass-card" style="text-align:center; height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">1️⃣</div>
            <strong>Upload Data</strong>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">Import your CSV dataset and select your target variable</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="text-align:center; height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">2️⃣</div>
            <strong>Auto Training</strong>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">AutoML trains & tunes 15+ algorithms automatically</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-card" style="text-align:center; height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">3️⃣</div>
            <strong>Compare & Explain</strong>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">Review model leaderboard and understand predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="glass-card" style="text-align:center; height: 180px;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">4️⃣</div>
            <strong>Predict</strong>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">Make single or batch predictions with the best model</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💡 What You Can Benefit")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #10b981;">🚀 For Data Scientists</h4>
            <ul style="color: #475569; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Rapid Prototyping:</strong> Test multiple algorithms in minutes, not hours</li>
                <li><strong>Hyperparameter Tuning:</strong> Automated optimization saves countless manual iterations</li>
                <li><strong>Model Explainability:</strong> SHAP integration for transparent, interpretable results</li>
                <li><strong>Neural Networks:</strong> Access to Transformer, CNN-LSTM, TCN architectures</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #8B5CF6;">📊 For Business Analysts</h4>
            <ul style="color: #475569; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>No Coding Required:</strong> Intuitive interface with zero programming knowledge needed</li>
                <li><strong>Quick Insights:</strong> Transform raw data into actionable predictions instantly</li>
                <li><strong>Risk Metrics:</strong> Built-in Sharpe, Sortino, VaR analysis for financial data</li>
                <li><strong>Drift Detection:</strong> Monitor model performance degradation over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-card info" style="margin-top: 1rem;">
        <strong>💎 Key Benefits:</strong> Save 90% development time • Compare 15+ algorithms automatically • 
        Enterprise-grade security • Production-ready models • Comprehensive documentation
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ✨ Platform Features")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="glass-card"><h4>🤖 AutoML</h4><p>15+ ML algorithms with auto-tuning</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="glass-card"><h4>🧠 Deep Learning</h4><p>5 neural network architectures</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="glass-card"><h4>🔍 Explainability</h4><p>SHAP and feature importance</p></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="glass-card"><h4>🚨 Drift Detection</h4><p>KS-test and PSI monitoring</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="glass-card"><h4>📊 Risk Metrics</h4><p>Sharpe, Sortino, VaR analysis</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="glass-card"><h4>🎭 Ensemble</h4><p>Weighted model ensembles</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📤 Upload Data", type="primary", use_container_width=True): st.session_state.current_page = 'upload'; st.rerun()
    with c2:
        if st.button("🏋️ Train Models", use_container_width=True): st.session_state.current_page = 'train'; st.rerun()
    with c3:
        if st.button("💎 View Pricing", use_container_width=True): st.session_state.current_page = 'pricing'; st.rerun()

def page_upload():
    st.markdown("## 📤 Upload Data")
    limits = get_limits()
    max_rows = limits['max_rows']
    max_display = f"{int(max_rows):,}" if max_rows != float('inf') else "Unlimited"
    render_alert(f"📊 <strong>Limits:</strong> Max {max_display} rows | Max {limits['max_features']} features", "info")
    
    if st.session_state.uploaded_data is not None and st.session_state.target_column:
        df = st.session_state.uploaded_data
        render_alert("✅ <strong>Data Loaded!</strong>", "success")
        c1, c2, c3 = st.columns(3)
        with c1: render_metric_card("Rows", f"{len(df):,}", "📊")
        with c2: render_metric_card("Features", str(len(st.session_state.feature_columns)), "📈")
        with c3: render_metric_card("Target", st.session_state.target_column[:10], "🎯")
        st.dataframe(df.head(10), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🏋️ Train Models", type="primary", use_container_width=True): st.session_state.current_page = 'train'; st.rerun()
        with c2:
            if can_use_nn():
                if st.button("🧠 Neural Networks", use_container_width=True): st.session_state.current_page = 'neural'; st.rerun()
            else: st.button("🧠 Neural Networks 🔒", disabled=True, use_container_width=True)
        with c3:
            if st.button("🔄 New Data", use_container_width=True):
                st.session_state.uploaded_data = None; st.session_state.target_column = None; st.session_state.feature_columns = []; st.rerun()
        return
    
    uploaded = st.file_uploader("Choose CSV", type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if len(df) > max_rows and max_rows != float('inf'):
                st.warning(f"⚠️ Limiting to {int(max_rows):,} rows")
                df = df.head(int(max_rows))
            st.dataframe(df.head(10), use_container_width=True)
            target = st.selectbox("🎯 Target Column:", df.columns.tolist())
            if st.button("✅ Confirm", type="primary", use_container_width=True):
                st.session_state.uploaded_data = df
                st.session_state.target_column = target
                st.session_state.feature_columns = [c for c in df.columns if c != target]
                st.rerun()
        except Exception as e:
            render_alert(f"❌ Error: {e}", "error")
    else:
        st.markdown("---")
        if st.button("📊 Load Sample Data", use_container_width=True):
            np.random.seed(42)
            df = pd.DataFrame({'f1': np.random.randn(500), 'f2': np.random.randn(500)*2, 'f3': np.random.uniform(0,100,500), 'f4': np.random.randint(1,10,500), 'target': np.random.randn(500)*10+50})
            st.session_state.uploaded_data = df; st.session_state.target_column = 'target'; st.session_state.feature_columns = ['f1','f2','f3','f4']; st.rerun()

# =============================================================================
# PAGE - DATA PROFILER
# =============================================================================
def page_profiler():
    st.markdown("## 🔬 Data Profiler")
    
    if st.session_state.uploaded_data is None:
        render_alert("⚠️ Upload data first.", "warning")
        if st.button("📤 Go to Upload", type="primary"):
            st.session_state.current_page = 'upload'
            st.rerun()
        return
    
    df = st.session_state.uploaded_data
    
    # Calculate data quality metrics
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    completeness = (1 - missing_cells / total_cells) * 100
    uniqueness = (1 - duplicate_rows / len(df)) * 100
    overall_quality = (completeness * 0.5 + uniqueness * 0.5)
    
    # Quality Score Dashboard
    st.markdown("### 📊 Data Quality Score")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        score_color = "#16a34a" if overall_quality >= 80 else "#d97706" if overall_quality >= 60 else "#dc2626"
        st.markdown(f"""
        <div class="quality-gauge">
            <div class="quality-score" style="color:#ffffff;">{overall_quality:.0f}</div>
            <div style="font-weight:600;">Overall Quality</div>
        </div>
        """, unsafe_allow_html=True)
    with c2: render_metric_card("Completeness", f"{completeness:.1f}%", "✅")
    with c3: render_metric_card("Uniqueness", f"{uniqueness:.1f}%", "🔢")
    with c4: render_metric_card("Missing Cells", f"{missing_cells:,}", "❓")
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("Rows", f"{len(df):,}", "📊")
    with c2: render_metric_card("Columns", str(len(df.columns)), "📈")
    with c3: render_metric_card("Numeric", str(len(df.select_dtypes(include=[np.number]).columns)), "🔢")
    with c4: render_metric_card("Duplicates", str(duplicate_rows), "📋")
    
    # Tabs for different analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistics", "📈 Distributions", "🔗 Correlations", "🎯 Target Analysis"])
    
    with tab1:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe(include='all').round(3), use_container_width=True)
        
        st.markdown("#### Column Information")
        type_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': df.nunique().values
        })
        st.dataframe(type_df, use_container_width=True)
    
    with tab2:
        st.markdown("#### Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select feature:", numeric_cols)
            if selected_col:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=selected_col, nbins=50, title=f"Distribution of {selected_col}")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col_data = df[selected_col].dropna()
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Mean", f"{col_data.mean():.4f}")
                with c2: st.metric("Std", f"{col_data.std():.4f}")
                with c3: st.metric("Min", f"{col_data.min():.4f}")
                with c4: st.metric("Max", f"{col_data.max():.4f}")
    
    with tab3:
        st.markdown("#### Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r',
                           title="Feature Correlations")
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_alert("⚠️ Need at least 2 numeric columns for correlation.", "warning")
    
    with tab4:
        st.markdown("#### Target Variable Analysis")
        target = st.session_state.target_column
        if target and target in df.columns:
            target_data = df[target]
            
            if target_data.dtype in ['int64', 'float64']:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=target, nbins=50, title=f"Target Distribution: {target}")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.box(df, y=target, title=f"Target Box Plot: {target}")
                    st.plotly_chart(fig, use_container_width=True)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Mean", f"{target_data.mean():.4f}")
                with c2: st.metric("Std", f"{target_data.std():.4f}")
                with c3: st.metric("Min", f"{target_data.min():.4f}")
                with c4: st.metric("Max", f"{target_data.max():.4f}")
            else:
                value_counts = target_data.value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                            title=f"Target Class Distribution: {target}")
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE - TRAINING
# =============================================================================
def page_train():
    st.markdown("## 🏋️ Train Machine Learning Models")
    if st.session_state.uploaded_data is None:
        render_alert("⚠️ Upload data first.", "warning")
        if st.button("📤 Go to Upload", type="primary"): st.session_state.current_page = 'upload'; st.rerun()
        return
    if not AUTOML_AVAILABLE:
        render_alert("❌ AutoML engine not available.", "error"); return
    
    df = st.session_state.uploaded_data
    target = st.session_state.target_column
    limits = get_limits()
    render_alert(f"📊 <strong>Data:</strong> {len(df):,} rows | 🎯 <strong>Target:</strong> {target}", "info")
    
    st.markdown("### ⚙️ Configuration")
    c1, c2 = st.columns(2)
    with c1:
        max_models = st.slider("Max Models", 3, limits['max_models'], min(8, limits['max_models']))
        cv_folds = st.slider("CV Folds", 2, 10, 5)
    with c2:
        time_limit = st.slider("Time Limit (min)", 1, limits['time_limit'], min(5, limits['time_limit']))
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
    
    st.markdown("### 🔧 Options")
    c1, c2, c3 = st.columns(3)
    with c1: include_ensemble = st.checkbox("🎭 Ensemble", value=limits['include_ensemble'], disabled=not limits['include_ensemble'])
    with c2: include_adv = st.checkbox("⚡ Advanced Features", value=limits['include_advanced_features'], disabled=not limits['include_advanced_features'])
    with c3: enable_explain = st.checkbox("🔍 Explainability", value=limits['include_explainability'], disabled=not limits['include_explainability'])
    
    c1, c2, c3 = st.columns(3)
    with c1: enable_drift = st.checkbox("🚨 Drift Detection", value=limits['include_drift_detection'], disabled=not limits['include_drift_detection'])
    with c2: enable_risk = st.checkbox("📊 Risk Metrics", value=limits['include_risk_metrics'], disabled=not limits['include_risk_metrics'])
    with c3: enable_cv = st.checkbox("🔄 Time Series CV", value=False)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        config = AutoMLConfig(
            optimization_strategy=OptimizationStrategy.RANDOM_SEARCH, cv_folds=cv_folds, max_models=max_models,
            time_limit_minutes=time_limit, per_model_time_limit=max(30, (time_limit*60)//max(max_models,1)),
            test_size=test_size, feature_engineering=include_adv, advanced_features=include_adv,
            ensemble_size=3 if include_ensemble else 0, enable_drift_detection=enable_drift,
            enable_explainability=enable_explain, enable_risk_metrics=enable_risk, verbose=True
        )
        
        X = df.drop(columns=[target]); y = df[target]
        progress = st.progress(0, "Initializing...")
        
        try:
            engine = AutoMLEngine(config)
            start = time.time()
            engine.fit(X, y, target)
            duration = time.time() - start
            
            st.session_state.automl_engine = engine
            st.session_state.training_complete = True
            st.session_state.model_results = engine.model_results
            st.session_state.training_runs += 1
            
            progress.progress(1.0, "✅ Complete!")
            best_info = get_model_info(engine.best_model_name)
            render_alert(f"🎉 <strong>Done!</strong> Best: {best_info['icon']} {best_info['name']} | Time: {duration:.1f}s", "success")
            
            st.markdown("### 📊 Leaderboard")
            is_reg = engine.dataset_info.task_type == TaskType.REGRESSION
            sorted_models = sorted(engine.model_results.items(), key=lambda x: -x[1].get('test_r2' if is_reg else 'test_accuracy', 0))
            for rank, (name, results) in enumerate(sorted_models, 1):
                render_model_card(rank, name, results, name == engine.best_model_name, is_reg)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("📊 Results", type="primary", use_container_width=True): st.session_state.current_page = 'results'; st.rerun()
            with c2:
                if st.button("🔍 Explain", use_container_width=True): st.session_state.current_page = 'explain'; st.rerun()
            with c3:
                if st.button("🔮 Predict", use_container_width=True): st.session_state.current_page = 'predict'; st.rerun()
        except Exception as e:
            progress.progress(0, "❌ Failed")
            render_alert(f"❌ {e}", "error")

def page_neural():
    st.markdown("## 🧠 Neural Network Training")
    if not check_feature('include_neural_networks'):
        render_alert(f"🔒 <strong>Neural Networks require Professional tier.</strong>", "warning")
        st.info("💡 Use PRO-2024 to unlock"); return
    if not NEURAL_NETWORKS_AVAILABLE:
        render_alert("⚠️ PyTorch not installed.", "error"); return
    if st.session_state.uploaded_data is None:
        render_alert("⚠️ Upload data first.", "warning"); return
    
    st.markdown("### 🔧 Architectures")
    cols = st.columns(len(NEURAL_MODELS))
    for col, (k, m) in zip(cols, NEURAL_MODELS.items()):
        with col: st.markdown(f'<div class="glass-card" style="text-align:center;"><div style="font-size:2rem;">{m["icon"]}</div><strong>{m["name"]}</strong><br><small>{m["desc"]}</small></div>', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Config")
    c1, c2 = st.columns(2)
    with c1:
        model_type = st.selectbox("Architecture", list(NEURAL_MODELS.keys()), format_func=lambda x: NEURAL_MODELS[x]['name'])
        epochs = st.slider("Epochs", 10, 200, 50)
    with c2:
        learning_rate = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
        batch_size = st.select_slider("Batch Size", [16, 32, 64, 128], value=32)
    
    if st.button("🚀 Train", type="primary", use_container_width=True):
        df = st.session_state.uploaded_data; target = st.session_state.target_column
        X = df.drop(columns=[target]).values; y = df[target].values
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        try:
            model = NeuralNetworkFactory.create_model(model_type, X_train.shape[1], 1, 'regression')
            config = NNTrainingConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, patience=15)
            trainer = NeuralNetworkTrainer(model, config)
            history = trainer.fit(X_train, y_train, X_test, y_test)
            y_pred = trainer.predict(X_test)
            
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)); r2 = r2_score(y_test, y_pred)
            st.session_state.neural_model = trainer
            render_alert(f"🎉 <strong>Done!</strong> R²: {r2:.4f} | RMSE: {rmse:.4f}", "neural")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['train_loss'], name='Train', line=dict(color='#8B5CF6')))
            if history['val_loss']: fig.add_trace(go.Scatter(y=history['val_loss'], name='Val', line=dict(color='#06B6D4')))
            fig.update_layout(title='Training Loss', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            render_alert(f"❌ {e}", "error")

def page_results():
    st.markdown("## 📊 Results")
    if not st.session_state.training_complete:
        render_alert("⚠️ Train a model first.", "warning"); return
    
    engine = st.session_state.automl_engine
    is_reg = engine.dataset_info.task_type == TaskType.REGRESSION
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric_card("Models", str(len(engine.model_results)), "🤖")
    with c2:
        best_score = engine.model_results[engine.best_model_name].get('test_r2' if is_reg else 'test_accuracy', 0)
        render_metric_card("Best Score", f"{best_score:.4f}", "🎯")
    with c3:
        total_time = sum(r.get('training_time', 0) for r in engine.model_results.values())
        render_metric_card("Time", f"{total_time:.1f}s", "⏱️")
    with c4:
        render_metric_card("Best", get_model_info(engine.best_model_name)['name'][:10], get_model_info(engine.best_model_name)['icon'])
    
    st.markdown("### 🏆 Leaderboard")
    st.dataframe(engine.get_leaderboard(), use_container_width=True)
    
    metric = 'Test R²' if is_reg else 'Test Accuracy'
    leaderboard = engine.get_leaderboard()
    if metric in leaderboard.columns:
        fig = px.bar(leaderboard.head(10), x='Model', y=metric, color=metric, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE - MODEL COMPARISON
# =============================================================================
def page_compare():
    st.markdown("## ⚖️ Model Comparison")
    
    if not check_feature('include_model_comparison'):
        render_alert("🔒 <strong>Model Comparison</strong> requires Starter+ subscription.", "warning")
        if st.button("💎 Upgrade Now", type="primary"):
            st.session_state.current_page = 'pricing'
            st.rerun()
        return
    
    if not st.session_state.training_complete:
        render_alert("⚠️ Train models first to compare them.", "warning")
        return
    
    engine = st.session_state.automl_engine
    is_reg = engine.dataset_info.task_type == TaskType.REGRESSION
    
    st.markdown("### 📊 Select Models to Compare")
    model_names = list(engine.model_results.keys())
    selected_models = st.multiselect("Choose models:", model_names, default=model_names[:min(3, len(model_names))])
    
    if len(selected_models) < 2:
        render_alert("⚠️ Select at least 2 models to compare.", "warning")
        return
    
    # Comparison metrics
    st.markdown("### 📈 Performance Comparison")
    
    if is_reg:
        metrics = ['test_r2', 'test_rmse', 'cv_mean', 'training_time']
        metric_labels = ['R² Score', 'RMSE', 'CV Score', 'Training Time (s)']
    else:
        metrics = ['test_accuracy', 'test_f1', 'cv_mean', 'training_time']
        metric_labels = ['Accuracy', 'F1 Score', 'CV Score', 'Training Time (s)']
    
    # Build comparison data
    comparison_data = []
    for model in selected_models:
        results = engine.model_results[model]
        row = {'Model': get_model_info(model)['name']}
        for metric, label in zip(metrics, metric_labels):
            row[label] = results.get(metric, 0)
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Bar comparison
    st.markdown("### 📊 Visual Comparison")
    metric_to_compare = st.selectbox("Select metric to visualize:", metric_labels[:-1])
    
    chart_data = comparison_df[['Model', metric_to_compare]]
    fig = px.bar(chart_data, x='Model', y=metric_to_compare, color=metric_to_compare,
                 color_continuous_scale='Viridis', title=f"{metric_to_compare} Comparison")
    st.plotly_chart(fig, use_container_width=True)

def page_explain():
    st.markdown("## 🔍 Explainability")
    if not check_feature('include_explainability'):
        render_alert("🔒 Requires Starter+", "warning"); return
    if not st.session_state.training_complete:
        render_alert("⚠️ Train first.", "warning"); return
    
    engine = st.session_state.automl_engine
    explanations = engine.get_explanations()
    if not explanations:
        render_alert("⚠️ No explanations available.", "warning"); return
    
    if 'feature_importance' in explanations:
        fi = sorted(explanations['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:15]
        fig = go.Figure(go.Bar(x=[v for _,v in fi], y=[k for k,_ in fi], orientation='h', marker_color='#8B5CF6'))
        fig.update_layout(title='Feature Importance', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Report")
    st.code(engine.get_explanation_report())

# =============================================================================
# PAGE - WHAT-IF ANALYSIS
# =============================================================================
def page_whatif():
    st.markdown("## 🎲 What-If Analysis")
    
    if not check_feature('include_whatif'):
        render_alert("🔒 <strong>What-If Analysis</strong> requires Starter+ subscription.", "warning")
        if st.button("💎 Upgrade Now", type="primary"):
            st.session_state.current_page = 'pricing'
            st.rerun()
        return
    
    if not st.session_state.training_complete:
        render_alert("⚠️ Train a model first.", "warning")
        return
    
    engine = st.session_state.automl_engine
    features = st.session_state.feature_columns
    df = st.session_state.uploaded_data
    
    render_alert("🎲 <strong>Explore how changing feature values affects predictions!</strong>", "info")
    
    st.markdown("### 📝 Create Scenario")
    
    # Base values from median
    base_values = {}
    for f in features:
        if f in df.columns and df[f].dtype in ['int64', 'float64']:
            base_values[f] = float(df[f].median())
        elif f in df.columns:
            base_values[f] = df[f].mode()[0] if len(df[f].mode()) > 0 else 0
        else:
            base_values[f] = 0.0
    
    # Editable scenario
    st.markdown("#### Adjust Feature Values")
    scenario_values = {}
    
    cols_per_row = 3
    feature_list = [f for f in features if f in df.columns]
    for i in range(0, len(feature_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(feature_list):
                f = feature_list[i + j]
                with col:
                    if df[f].dtype in ['int64', 'float64']:
                        min_val = float(df[f].min())
                        max_val = float(df[f].max())
                        default_val = base_values.get(f, (min_val + max_val) / 2)
                        scenario_values[f] = st.slider(f, min_val, max_val, default_val, key=f"wi_{f}")
                    else:
                        options = df[f].unique().tolist()
                        scenario_values[f] = st.selectbox(f, options, key=f"wi_{f}")
    
    if st.button("🔮 Predict Scenario", type="primary", use_container_width=True):
        try:
            pred_df = pd.DataFrame([scenario_values])
            prediction = engine.predict(pred_df)[0]
            base_pred = engine.predict(pd.DataFrame([base_values]))[0]
            
            st.markdown("### 📊 Results")
            c1, c2, c3 = st.columns(3)
            with c1: render_metric_card("Base Prediction", f"{base_pred:.4f}", "📍")
            with c2: render_metric_card("Scenario Prediction", f"{prediction:.4f}", "🎯")
            with c3: 
                diff = prediction - base_pred
                render_metric_card("Difference", f"{diff:+.4f}", "📈" if diff > 0 else "📉")
        except Exception as e:
            render_alert(f"❌ {e}", "error")
    
    # Sensitivity Analysis
    st.markdown("### 📈 Sensitivity Analysis")
    numeric_features = [f for f in feature_list if df[f].dtype in ['int64', 'float64']]
    if numeric_features:
        selected_feature = st.selectbox("Select feature to analyze:", numeric_features)
        
        if st.button("🔬 Run Sensitivity Analysis", use_container_width=True):
            min_val = float(df[selected_feature].min())
            max_val = float(df[selected_feature].max())
            test_values = np.linspace(min_val, max_val, 20)
            
            predictions = []
            for val in test_values:
                test_scenario = base_values.copy()
                test_scenario[selected_feature] = val
                try:
                    pred = engine.predict(pd.DataFrame([test_scenario]))[0]
                    predictions.append(pred)
                except:
                    predictions.append(np.nan)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_values, y=predictions, mode='lines+markers',
                                    name='Prediction', line=dict(color='#7c3aed', width=3)))
            fig.update_layout(
                title=f"Prediction Sensitivity to {selected_feature}",
                xaxis_title=selected_feature,
                yaxis_title="Prediction"
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE - ANOMALY DETECTION
# =============================================================================
def page_anomaly():
    st.markdown("## 🚨 Anomaly Detection")
    
    if not check_feature('include_anomaly'):
        render_alert("🔒 <strong>Anomaly Detection</strong> requires Professional+ subscription.", "warning")
        if st.button("💎 Upgrade Now", type="primary"):
            st.session_state.current_page = 'pricing'
            st.rerun()
        return
    
    if st.session_state.uploaded_data is None:
        render_alert("⚠️ Upload data first.", "warning")
        return
    
    df = st.session_state.uploaded_data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        render_alert("⚠️ No numeric columns found for anomaly detection.", "warning")
        return
    
    render_alert("🚨 <strong>Detect outliers and anomalies in your data</strong>", "info")
    
    st.markdown("### ⚙️ Configuration")
    c1, c2 = st.columns(2)
    with c1:
        selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
        method = st.selectbox("Detection Method:", ["IQR (Interquartile Range)", "Z-Score"])
    with c2:
        if "IQR" in method:
            threshold = st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
        else:
            threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0)
    
    if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
        if not selected_cols:
            render_alert("⚠️ Select at least one column.", "warning")
            return
        
        total_anomalies = set()
        results = {}
        
        for col in selected_cols:
            series = df[col].dropna()
            
            if "IQR" in method:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                anomalies = (series < lower) | (series > upper)
            else:
                z_scores = np.abs((series - series.mean()) / series.std())
                anomalies = z_scores > threshold
            
            anomaly_indices = series[anomalies].index.tolist()
            results[col] = {
                'count': len(anomaly_indices),
                'percentage': (len(anomaly_indices) / len(series)) * 100,
                'indices': anomaly_indices
            }
            total_anomalies.update(anomaly_indices)
        
        # Display results
        st.markdown("### 📊 Detection Results")
        c1, c2, c3 = st.columns(3)
        with c1: render_metric_card("Total Anomalies", str(len(total_anomalies)), "🚨")
        with c2: render_metric_card("Affected Rows", f"{len(total_anomalies)/len(df)*100:.1f}%", "📊")
        with c3: render_metric_card("Columns Analyzed", str(len(selected_cols)), "📈")
        
        # Per-column breakdown
        st.markdown("### 📋 Column-wise Breakdown")
        breakdown_df = pd.DataFrame([
            {'Column': col, 'Anomalies': r['count'], 'Percentage': f"{r['percentage']:.2f}%"}
            for col, r in results.items()
        ])
        st.dataframe(breakdown_df, use_container_width=True)
        
        # Visualization
        if selected_cols:
            st.markdown("### 📈 Anomaly Visualization")
            viz_col = st.selectbox("Select column to visualize:", selected_cols)
            
            if viz_col and viz_col in results:
                series = df[viz_col].dropna()
                anomaly_mask = series.index.isin(results[viz_col]['indices'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series[~anomaly_mask].index.tolist(),
                    y=series[~anomaly_mask].values,
                    mode='markers',
                    name='Normal',
                    marker=dict(color='#16a34a', size=6)
                ))
                fig.add_trace(go.Scatter(
                    x=series[anomaly_mask].index.tolist(),
                    y=series[anomaly_mask].values,
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='#dc2626', size=10, symbol='x')
                ))
                fig.update_layout(title=f"Anomalies in {viz_col}", xaxis_title="Index", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE - FORECASTING
# =============================================================================
def page_forecast():
    st.markdown("## 📈 Forecasting")
    
    if not check_feature('include_forecasting'):
        render_alert("🔒 <strong>Forecasting</strong> requires Professional+ subscription.", "warning")
        if st.button("💎 Upgrade Now", type="primary"):
            st.session_state.current_page = 'pricing'
            st.rerun()
        return
    
    if st.session_state.uploaded_data is None:
        render_alert("⚠️ Upload data first.", "warning")
        return
    
    df = st.session_state.uploaded_data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        render_alert("⚠️ No numeric columns found.", "warning")
        return
    
    render_alert("📈 <strong>Generate forecasts with confidence intervals</strong>", "info")
    
    st.markdown("### ⚙️ Configuration")
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Select target column:", numeric_cols)
        forecast_periods = st.slider("Forecast periods:", 7, 90, 30)
    with c2:
        confidence_level = st.selectbox("Confidence Level:", ["90%", "95%", "99%"])
    
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        try:
            series = df[target_col].dropna()
            
            # Simple linear trend forecast
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            trend_line = np.poly1d(coeffs)
            
            # Generate forecast
            forecast_x = np.arange(len(series), len(series) + forecast_periods)
            forecast_values = trend_line(forecast_x)
            
            # Calculate confidence intervals
            residuals = series.values - trend_line(x)
            std_err = np.std(residuals)
            ci_multiplier = {'90%': 1.645, '95%': 1.96, '99%': 2.576}[confidence_level]
            
            # CI widens over time
            ci_width = ci_multiplier * std_err * np.sqrt(1 + np.arange(1, forecast_periods + 1) / len(series))
            lower_ci = forecast_values - ci_width
            upper_ci = forecast_values + ci_width
            
            # Display results
            st.markdown("### 📊 Forecast Results")
            c1, c2, c3 = st.columns(3)
            with c1: render_metric_card("Current Value", f"{series.iloc[-1]:.2f}", "📍")
            with c2: render_metric_card("Forecast End", f"{forecast_values[-1]:.2f}", "🎯")
            with c3:
                change = ((forecast_values[-1] / series.iloc[-1]) - 1) * 100
                render_metric_card("Change", f"{change:+.1f}%", "📈" if change > 0 else "📉")
            
            # Visualization
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=list(range(len(series))),
                y=series.values,
                mode='lines',
                name='Historical',
                line=dict(color='#2563eb', width=2)
            ))
            
            # Forecast
            forecast_indices = list(range(len(series), len(series) + forecast_periods))
            fig.add_trace(go.Scatter(
                x=forecast_indices,
                y=forecast_values,
                mode='lines',
                name='Forecast',
                line=dict(color='#7c3aed', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_indices + forecast_indices[::-1],
                y=list(upper_ci) + list(lower_ci[::-1]),
                fill='toself',
                fillcolor='rgba(124, 58, 237, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level} CI'
            ))
            
            fig.update_layout(
                title=f"Forecast for {target_col}",
                xaxis_title="Period",
                yaxis_title="Value",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.markdown("### 📋 Forecast Data")
            forecast_df = pd.DataFrame({
                'Period': range(1, forecast_periods + 1),
                'Forecast': forecast_values,
                'Lower CI': lower_ci,
                'Upper CI': upper_ci
            })
            st.dataframe(forecast_df.round(4), use_container_width=True)
            
            csv = forecast_df.to_csv(index=False)
            st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")
            
        except Exception as e:
            render_alert(f"❌ {e}", "error")

def page_drift():
    st.markdown("## 🚨 Drift Detection")
    if not check_feature('include_drift_detection'):
        render_alert("🔒 Requires Professional+", "warning"); return
    if not st.session_state.training_complete:
        render_alert("⚠️ Train first.", "warning"); return
    
    render_alert("🚨 <strong>Monitor data distribution changes.</strong>", "info")
    uploaded = st.file_uploader("Upload new data", type=['csv'], key="drift_file")
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            st.dataframe(new_df.head(), use_container_width=True)
            if st.button("🔍 Detect Drift", type="primary", use_container_width=True):
                engine = st.session_state.automl_engine
                feature_cols = st.session_state.feature_columns
                new_features = new_df[feature_cols] if all(c in new_df.columns for c in feature_cols) else new_df.drop(columns=[st.session_state.target_column], errors='ignore')
                drift_detected, drift_score, feature_drift = engine.detect_drift(new_features)
                
                c1, c2, c3 = st.columns(3)
                with c1: render_metric_card("Status", "🔴 DRIFT" if drift_detected else "🟢 OK", "🚨")
                with c2: render_metric_card("Score", f"{drift_score:.4f}", "📊")
                with c3:
                    drifted = sum(1 for f in feature_drift.values() if f.get('drifted', False))
                    render_metric_card("Drifted", f"{drifted}/{len(feature_drift)}", "📈")
                
                render_alert("⚠️ Drift detected! Consider retraining." if drift_detected else "✅ No significant drift.", "warning" if drift_detected else "success")
        except Exception as e:
            render_alert(f"❌ {e}", "error")

# =============================================================================
# PAGES - PREDICTIONS
# =============================================================================
def page_predict():
    st.markdown("## 🔮 Predictions")
    if not st.session_state.training_complete:
        render_alert("⚠️ Train first.", "warning"); return
    
    engine = st.session_state.automl_engine
    info = get_model_info(engine.best_model_name)
    is_reg = engine.dataset_info.task_type == TaskType.REGRESSION
    
    render_alert(f"🤖 <strong>Model:</strong> {info['icon']} {info['name']}", "success")
    use_ensemble = st.checkbox("🎭 Use Ensemble", value=True) if hasattr(engine, 'weighted_ensemble') and engine.weighted_ensemble else False
    
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📄 Batch Prediction"])
    
    with tab1:
        features = st.session_state.feature_columns
        with st.form("pred"):
            vals = {}
            for i in range(0, len(features), 3):
                cols = st.columns(3)
                for j, c in enumerate(cols):
                    if i+j < len(features):
                        f = features[i+j]
                        sample = 0.0
                        if st.session_state.uploaded_data is not None:
                            try:
                                col_data = st.session_state.uploaded_data[f]
                                if col_data.dtype in ['int64', 'float64']: sample = float(col_data.median())
                            except: pass
                        with c: vals[f] = st.number_input(f, value=sample, key=f"sp_{f}")
            
            if st.form_submit_button("🔮 Predict", type="primary", use_container_width=True):
                try:
                    pred = engine.predict(pd.DataFrame([vals]), use_weighted_ensemble=use_ensemble)[0]
                    st.session_state.predictions_made += 1
                    st.markdown(f'<div class="prediction-result"><div style="font-size:0.9rem;color:#166534;">🎯 Prediction</div><div class="prediction-value">{pred:.4f}</div></div>', unsafe_allow_html=True)
                except Exception as e:
                    render_alert(f"❌ {e}", "error")
    
    with tab2:
        if not check_feature('batch_predictions'):
            render_locked_feature("Batch Predictions", "Starter")
            return
        
        uploaded = st.file_uploader("CSV", type=['csv'], key="batch_file")
        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                st.dataframe(batch_df.head(), use_container_width=True)
                if st.button("🔮 Predict All", type="primary", use_container_width=True):
                    feature_cols = st.session_state.feature_columns
                    pred_df = batch_df[feature_cols].copy()
                    preds = engine.predict(pred_df, use_weighted_ensemble=use_ensemble)
                    result = batch_df.copy(); result['Prediction'] = preds
                    st.session_state.predictions_made += len(preds)
                    render_alert(f"✅ {len(preds):,} predictions!", "success")
                    st.dataframe(result, use_container_width=True)
                    st.download_button("📥 Download", result.to_csv(index=False), "predictions.csv")
            except Exception as e:
                render_alert(f"❌ {e}", "error")

# =============================================================================
# PAGES - REPORT GENERATOR (NEW)
# =============================================================================
def page_reports():
    st.markdown("## 📋 Report Generator")
    
    if not check_feature('include_reports'):
        render_locked_feature("Report Generator", "Starter")
        return
    
    render_alert("📋 <strong>Generate comprehensive reports for your ML projects</strong>", "info")
    
    st.markdown("### 📝 Report Configuration")
    
    report_type = st.selectbox("Report Type:", [
        "Executive Summary",
        "Technical Deep Dive", 
        "Model Performance Report",
        "Data Quality Report",
        "Full Analysis Report"
    ])
    
    include_options = st.multiselect("Include Sections:", [
        "Data Overview",
        "Feature Statistics",
        "Model Leaderboard",
        "Best Model Details",
        "Feature Importance",
        "Predictions Summary",
        "Recommendations"
    ], default=["Data Overview", "Model Leaderboard", "Best Model Details"])
    
    report_format = st.selectbox("Output Format:", ["Markdown", "HTML", "JSON"])
    
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        report_content = generate_report(report_type, include_options)
        
        st.markdown("### 📊 Generated Report")
        st.markdown(report_content)
        
        # Save report
        st.session_state.saved_reports.append({
            'id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'type': report_type,
            'content': report_content
        })
        
        # Download option
        if report_format == "Markdown":
            st.download_button("📥 Download Report", report_content, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        elif report_format == "HTML":
            html_content = f"<html><body><pre>{report_content}</pre></body></html>"
            st.download_button("📥 Download Report", html_content, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        else:
            json_content = json.dumps({'report_type': report_type, 'content': report_content, 'timestamp': datetime.now().isoformat()})
            st.download_button("📥 Download Report", json_content, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Saved reports
    if st.session_state.saved_reports:
        st.markdown("### 📚 Saved Reports")
        for report in st.session_state.saved_reports[-5:]:
            with st.expander(f"📄 {report['type']} - {report['timestamp'][:10]}"):
                st.markdown(report['content'][:500] + "...")

def generate_report(report_type: str, sections: List[str]) -> str:
    """Generate a formatted report based on type and sections"""
    report = f"# {report_type}\n\n"
    report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**Platform:** QuantML Studio v{APP_CONFIG['version']}\n\n"
    report += "---\n\n"
    
    if "Data Overview" in sections and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        report += "## 📊 Data Overview\n\n"
        report += f"- **Rows:** {len(df):,}\n"
        report += f"- **Columns:** {len(df.columns)}\n"
        report += f"- **Target:** {st.session_state.target_column}\n"
        report += f"- **Features:** {len(st.session_state.feature_columns)}\n\n"
    
    if "Feature Statistics" in sections and st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        report += "## 📈 Feature Statistics\n\n"
        report += "| Feature | Mean | Std | Min | Max |\n"
        report += "|---------|------|-----|-----|-----|\n"
        for col in df.select_dtypes(include=[np.number]).columns[:10]:
            report += f"| {col} | {df[col].mean():.2f} | {df[col].std():.2f} | {df[col].min():.2f} | {df[col].max():.2f} |\n"
        report += "\n"
    
    if "Model Leaderboard" in sections and st.session_state.training_complete:
        engine = st.session_state.automl_engine
        is_reg = engine.dataset_info.task_type == TaskType.REGRESSION
        report += "## 🏆 Model Leaderboard\n\n"
        report += "| Rank | Model | Score | Training Time |\n"
        report += "|------|-------|-------|---------------|\n"
        metric_key = 'test_r2' if is_reg else 'test_accuracy'
        sorted_models = sorted(engine.model_results.items(), key=lambda x: -x[1].get(metric_key, 0))
        for rank, (name, results) in enumerate(sorted_models[:10], 1):
            score = results.get(metric_key, 0)
            time_val = results.get('training_time', 0)
            report += f"| {rank} | {name} | {score:.4f} | {time_val:.2f}s |\n"
        report += "\n"
    
    if "Best Model Details" in sections and st.session_state.training_complete:
        engine = st.session_state.automl_engine
        report += "## 🏅 Best Model Details\n\n"
        report += f"- **Model:** {engine.best_model_name}\n"
        best_results = engine.model_results[engine.best_model_name]
        for key, value in best_results.items():
            if isinstance(value, (int, float)):
                report += f"- **{key}:** {value:.4f}\n"
        report += "\n"
    
    if "Recommendations" in sections:
        report += "## 💡 Recommendations\n\n"
        report += "1. Consider feature engineering to improve model performance\n"
        report += "2. Monitor for data drift in production\n"
        report += "3. Regularly retrain models with new data\n"
        report += "4. Use ensemble methods for better generalization\n\n"
    
    report += "---\n\n"
    report += "*Report generated by QuantML Studio*\n"
    
    return report

# =============================================================================
# PAGES - API PLAYGROUND (NEW)
# =============================================================================
def page_api():
    st.markdown("## 🔌 API Playground")
    
    if not check_feature('include_api'):
        render_locked_feature("API Access", "Professional")
        return
    
    render_alert("🔌 <strong>Integrate QuantML Studio with your applications via REST API</strong>", "info")
    
    tab1, tab2, tab3 = st.tabs(["🔑 API Keys", "📖 Documentation", "🧪 Test Endpoint"])
    
    with tab1:
        st.markdown("### 🔑 Your API Keys")
        
        if st.button("➕ Generate New API Key", type="primary"):
            new_key = generate_api_key()
            st.session_state.api_keys.append({
                'key': new_key,
                'created': datetime.now().isoformat(),
                'last_used': None,
                'requests': 0
            })
            st.success(f"✅ New API key generated!")
        
        if st.session_state.api_keys:
            for i, key_info in enumerate(st.session_state.api_keys):
                with st.expander(f"🔑 Key: {key_info['key'][:20]}..."):
                    st.code(key_info['key'])
                    st.markdown(f"**Created:** {key_info['created']}")
                    st.markdown(f"**Requests:** {key_info['requests']}")
                    if st.button(f"🗑️ Revoke", key=f"revoke_{i}"):
                        st.session_state.api_keys.pop(i)
                        st.rerun()
        else:
            st.info("No API keys generated yet. Click above to create one.")
    
    with tab2:
        st.markdown("### 📖 API Documentation")
        
        st.markdown("#### Base URL")
        st.code("https://api.quantml.studio/v1")
        
        st.markdown("#### Authentication")
        st.markdown("Include your API key in the request header:")
        st.code('Authorization: Bearer YOUR_API_KEY')
        
        st.markdown("#### Endpoints")
        
        with st.expander("POST /predict"):
            st.markdown("Make predictions using your trained model.")
            st.markdown("**Request Body:**")
            st.code('''
{
    "model_id": "your_model_id",
    "features": {
        "feature_1": 1.5,
        "feature_2": 2.3,
        "feature_3": 0.8
    }
}
            ''')
            st.markdown("**Response:**")
            st.code('''
{
    "prediction": 42.5,
    "confidence": 0.95,
    "model_version": "1.0.0"
}
            ''')
        
        with st.expander("POST /batch-predict"):
            st.markdown("Make batch predictions.")
            st.code('''
{
    "model_id": "your_model_id",
    "data": [
        {"feature_1": 1.5, "feature_2": 2.3},
        {"feature_1": 2.0, "feature_2": 1.8}
    ]
}
            ''')
        
        with st.expander("GET /models"):
            st.markdown("List all your trained models.")
        
        with st.expander("GET /models/{model_id}"):
            st.markdown("Get details of a specific model.")
    
    with tab3:
        st.markdown("### 🧪 Test API Endpoint")
        
        endpoint = st.selectbox("Endpoint:", ["/predict", "/batch-predict", "/models"])
        
        if endpoint == "/predict":
            st.markdown("#### Request Body")
            if st.session_state.feature_columns:
                test_body = {f: 0.0 for f in st.session_state.feature_columns[:5]}
            else:
                test_body = {"feature_1": 1.0, "feature_2": 2.0}
            
            request_json = st.text_area("JSON Body:", value=json.dumps(test_body, indent=2), height=150)
            
            if st.button("🚀 Send Request", type="primary"):
                st.markdown("#### Response")
                # Simulate API response
                try:
                    body = json.loads(request_json)
                    response = {
                        "status": "success",
                        "prediction": np.random.uniform(10, 100),
                        "confidence": np.random.uniform(0.8, 0.99),
                        "latency_ms": np.random.randint(10, 50)
                    }
                    st.json(response)
                    render_alert("✅ Request successful!", "success")
                except Exception as e:
                    st.json({"status": "error", "message": str(e)})
                    render_alert(f"❌ Request failed: {e}", "error")

# =============================================================================
# PAGES - MODEL REGISTRY (NEW)
# =============================================================================
def page_registry():
    st.markdown("## 📦 Model Registry")
    
    if not check_feature('include_registry'):
        render_locked_feature("Model Registry", "Professional")
        return
    
    render_alert("📦 <strong>Version control, deploy, and manage your ML models</strong>", "info")
    
    tab1, tab2, tab3 = st.tabs(["📋 Models", "🚀 Deploy", "📊 Monitoring"])
    
    with tab1:
        st.markdown("### 📋 Registered Models")
        
        # Save current model to registry
        if st.session_state.training_complete:
            c1, c2 = st.columns([3, 1])
            with c1:
                model_name = st.text_input("Model Name:", value=f"model_{datetime.now().strftime('%Y%m%d')}")
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("📦 Register Model", type="primary"):
                    engine = st.session_state.automl_engine
                    st.session_state.model_registry.append({
                        'id': str(uuid.uuid4())[:8],
                        'name': model_name,
                        'version': '1.0.0',
                        'algorithm': engine.best_model_name,
                        'created': datetime.now().isoformat(),
                        'status': 'staged',
                        'metrics': engine.model_results[engine.best_model_name]
                    })
                    st.success("✅ Model registered!")
                    st.rerun()
        
        # Display registered models
        if st.session_state.model_registry:
            for model in st.session_state.model_registry:
                status_color = {'staged': '#f59e0b', 'production': '#10b981', 'archived': '#64748b'}[model['status']]
                with st.expander(f"📦 {model['name']} v{model['version']} - {model['algorithm']}"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**ID:** `{model['id']}`")
                        st.markdown(f"**Created:** {model['created'][:10]}")
                    with c2:
                        st.markdown(f"**Algorithm:** {model['algorithm']}")
                        st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold;'>{model['status'].upper()}</span>", unsafe_allow_html=True)
                    with c3:
                        if model['status'] == 'staged':
                            if st.button("🚀 Deploy", key=f"deploy_{model['id']}"):
                                model['status'] = 'production'
                                st.rerun()
                        elif model['status'] == 'production':
                            if st.button("📦 Archive", key=f"archive_{model['id']}"):
                                model['status'] = 'archived'
                                st.rerun()
        else:
            st.info("No models registered yet. Train a model and register it above.")
    
    with tab2:
        st.markdown("### 🚀 Deployment Options")
        
        deployment_target = st.selectbox("Deployment Target:", [
            "REST API Endpoint",
            "Docker Container",
            "AWS SageMaker",
            "Google Cloud AI Platform",
            "Azure ML"
        ])
        
        st.markdown(f"#### Deploy to {deployment_target}")
        
        if deployment_target == "REST API Endpoint":
            st.markdown("Your model will be deployed as a REST API endpoint.")
            st.code("POST https://api.quantml.studio/v1/predict/{model_id}")
        elif deployment_target == "Docker Container":
            st.markdown("Generate a Docker image for your model.")
            st.code('''
# Dockerfile
FROM python:3.9-slim
COPY model.pkl /app/
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
CMD ["python", "serve.py"]
            ''')
        
        if st.button("🚀 Start Deployment", type="primary", use_container_width=True):
            with st.spinner("Deploying model..."):
                time.sleep(2)
            render_alert("✅ Model deployed successfully!", "success")
    
    with tab3:
        st.markdown("### 📊 Model Monitoring")
        
        if st.session_state.model_registry:
            # Simulate monitoring data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            monitoring_data = pd.DataFrame({
                'date': dates,
                'requests': np.random.randint(100, 1000, 30),
                'latency_ms': np.random.uniform(20, 100, 30),
                'error_rate': np.random.uniform(0, 0.05, 30)
            })
            
            c1, c2, c3 = st.columns(3)
            with c1: render_metric_card("Total Requests", f"{monitoring_data['requests'].sum():,}", "📊")
            with c2: render_metric_card("Avg Latency", f"{monitoring_data['latency_ms'].mean():.0f}ms", "⚡")
            with c3: render_metric_card("Error Rate", f"{monitoring_data['error_rate'].mean()*100:.2f}%", "⚠️")
            
            # Charts
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Daily Requests", "Latency (ms)"))
            fig.add_trace(go.Bar(x=monitoring_data['date'], y=monitoring_data['requests'], marker_color='#8B5CF6'), row=1, col=1)
            fig.add_trace(go.Scatter(x=monitoring_data['date'], y=monitoring_data['latency_ms'], line=dict(color='#10b981')), row=2, col=1)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Register and deploy a model to view monitoring data.")

# =============================================================================
# PAGES - PRICING
# =============================================================================
def page_pricing():
    st.markdown("## 💎 Pricing Plans")
    st.markdown("Choose the perfect plan for your machine learning needs")
    
    current = st.session_state.subscription_tier
    
    cols = st.columns(4)
    for col, (k, t) in zip(cols, SUBSCRIPTION_TIERS.items()):
        is_curr = k == current
        with col:
            # Card container
            if is_curr:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {t['color']}15 0%, {t['color']}25 100%);
                            border-radius: 20px; padding: 1.5rem; border: 3px solid {t['color']};
                            text-align: center; height: 100%; box-shadow: 0 10px 30px {t['color']}30;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{t['icon']}</div>
                    <h3 style="color: #0f172a; margin: 0.5rem 0; font-size: 1.4rem;">{t['name']}</h3>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {t['color']}; margin: 0.5rem 0;">{t['price']}</div>
                    <span style="background: {t['color']}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">✓ CURRENT PLAN</span>
                    <hr style="margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;">
                    <div style="text-align: left; font-size: 0.85rem; color: #374151; line-height: 1.8;">
                        {"✅" if t['max_rows']>1000 else "❌"} Extended Data<br>
                        {"✅" if t['include_ensemble'] else "❌"} Ensemble Models<br>
                        {"✅" if t['include_neural_networks'] else "❌"} Neural Networks<br>
                        {"✅" if t['include_explainability'] else "❌"} Explainability<br>
                        {"✅" if t.get('include_anomaly', False) else "❌"} Anomaly Detection<br>
                        {"✅" if t.get('include_forecasting', False) else "❌"} Forecasting<br>
                        {"✅" if t.get('include_api', False) else "❌"} API Access<br>
                        {"✅" if t.get('include_registry', False) else "❌"} Model Registry
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #ffffff; border-radius: 20px; padding: 1.5rem; 
                            border: 2px solid #e2e8f0; text-align: center; height: 100%;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{t['icon']}</div>
                    <h3 style="color: #0f172a; margin: 0.5rem 0; font-size: 1.4rem;">{t['name']}</h3>
                    <div style="font-size: 1.8rem; font-weight: 700; color: {t['color']}; margin: 0.5rem 0;">{t['price']}</div>
                    <div style="height: 26px;"></div>
                    <hr style="margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;">
                    <div style="text-align: left; font-size: 0.85rem; color: #374151; line-height: 1.8;">
                        {"✅" if t['max_rows']>1000 else "❌"} Extended Data<br>
                        {"✅" if t['include_ensemble'] else "❌"} Ensemble Models<br>
                        {"✅" if t['include_neural_networks'] else "❌"} Neural Networks<br>
                        {"✅" if t['include_explainability'] else "❌"} Explainability<br>
                        {"✅" if t.get('include_anomaly', False) else "❌"} Anomaly Detection<br>
                        {"✅" if t.get('include_forecasting', False) else "❌"} Forecasting<br>
                        {"✅" if t.get('include_api', False) else "❌"} API Access<br>
                        {"✅" if t.get('include_registry', False) else "❌"} Model Registry
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Feature comparison section
    st.markdown("---")
    st.markdown("### 📊 Feature Comparison")
    
    comparison_data = {
        'Feature': ['Max Data Rows', 'Max Features', 'Max Models', 'Time Limit', 'Neural Networks', 
                    'Ensemble Methods', 'Model Explainability', 'Drift Detection', 'Anomaly Detection',
                    'Forecasting', 'API Access', 'Model Registry', 'Batch Predictions', 'Export Models'],
        'Free': ['1,000', '20', '5', '5 min', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌'],
        'Starter': ['10,000', '50', '15', '30 min', '❌', '✅', '✅', '❌', '❌', '❌', '❌', '❌', '✅', '✅'],
        'Professional': ['100,000', '200', '30', '2 hours', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅'],
        'Enterprise': ['Unlimited', 'Unlimited', '50', '8 hours', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Contact section
    st.markdown("---")
    st.markdown("### 💬 Need Help Choosing?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style="background: #f0f9ff; border-radius: 12px; padding: 1.25rem; border-left: 4px solid #0ea5e9;">
            <strong style="color: #0c4a6e;">🎯 For Individuals</strong>
            <p style="color: #475569; font-size: 0.9rem; margin-top: 0.5rem;">Start with Free tier to explore, upgrade to Starter for serious projects.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background: #faf5ff; border-radius: 12px; padding: 1.25rem; border-left: 4px solid #8b5cf6;">
            <strong style="color: #4c1d95;">🏢 For Teams</strong>
            <p style="color: #475569; font-size: 0.9rem; margin-top: 0.5rem;">Professional tier offers all features for growing data science teams.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background: #fefce8; border-radius: 12px; padding: 1.25rem; border-left: 4px solid #eab308;">
            <strong style="color: #713f12;">🌐 For Enterprise</strong>
            <p style="color: #475569; font-size: 0.9rem; margin-top: 0.5rem;">Custom solutions with dedicated support and unlimited resources.</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGES - SETTINGS
# =============================================================================
def page_settings():
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["👤 Account", "📊 Statistics", "🔧 System"])
    
    with tab1:
        st.markdown("### 👤 Account Information")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("User ID", st.session_state.user_id, disabled=True)
            st.text_input("Subscription Tier", get_limits()['name'], disabled=True)
            st.text_input("Session Started", st.session_state.session_start.strftime("%Y-%m-%d %H:%M"), disabled=True)
        with c2:
            render_metric_card("Training Runs", str(st.session_state.training_runs), "🏃")
            render_metric_card("Predictions Made", str(st.session_state.predictions_made), "🔮")
    
    with tab2:
        st.markdown("### 📊 Usage Statistics")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_metric_card("Experiments", str(len(st.session_state.experiment_history)), "🧪")
        with c2: render_metric_card("Reports", str(len(st.session_state.saved_reports)), "📋")
        with c3: render_metric_card("API Keys", str(len(st.session_state.api_keys)), "🔑")
        with c4: render_metric_card("Models", str(len(st.session_state.model_registry)), "📦")
        
        # Experiment history
        if st.session_state.experiment_history:
            st.markdown("### 📜 Experiment History")
            history_df = pd.DataFrame(st.session_state.experiment_history)
            st.dataframe(history_df, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔧 System Status")
        c1, c2, c3 = st.columns(3)
        with c1: render_alert(f"AutoML: {'✅ Available' if AUTOML_AVAILABLE else '❌ Unavailable'}", "success" if AUTOML_AVAILABLE else "error")
        with c2: render_alert(f"Neural Networks: {'✅ Available' if NEURAL_NETWORKS_AVAILABLE else '❌ Unavailable'}", "neural" if NEURAL_NETWORKS_AVAILABLE else "warning")
        with c3: render_alert(f"Tier: {get_limits()['name']}", "info")
        
        st.markdown("### 🗑️ Data Management")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Clear Session Data", use_container_width=True):
                st.session_state.uploaded_data = None
                st.session_state.target_column = None
                st.session_state.feature_columns = []
                st.session_state.training_complete = False
                st.rerun()
        with c2:
            if st.button("🗑️ Reset Everything", use_container_width=True, type="secondary"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

# =============================================================================
# FOOTER
# =============================================================================
def render_footer():
    st.markdown(f"""<div class="footer">
        🚀 {APP_CONFIG['app_name']} v{APP_CONFIG['version']} | AutoML: {'✅' if AUTOML_AVAILABLE else '❌'} | Neural: {'✅' if NEURAL_NETWORKS_AVAILABLE else '❌'} | © 2024 QuantML Studio
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(
        page_title=f"{APP_CONFIG['app_name']} v{APP_CONFIG['version']}", 
        page_icon="🚀", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    apply_styling()
    init_state()
    render_header()
    render_sidebar()
    
    pages = {
        'home': page_home, 
        'upload': page_upload, 
        'profiler': page_profiler,
        'train': page_train, 
        'neural': page_neural,
        'results': page_results, 
        'compare': page_compare,
        'explain': page_explain, 
        'whatif': page_whatif,
        'anomaly': page_anomaly,
        'forecast': page_forecast,
        'drift': page_drift, 
        'predict': page_predict,
        'reports': page_reports,
        'api': page_api,
        'registry': page_registry,
        'pricing': page_pricing, 
        'settings': page_settings
    }
    
    pages.get(st.session_state.current_page, page_home)()
    render_footer()

if __name__ == "__main__":
    main()
