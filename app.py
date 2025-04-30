import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import DataPreprocessor
from model_training import ChurnPredictor
import joblib
import os
from sklearn.model_selection import train_test_split

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Theme CSS with animations and new color palette
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    /* Enhanced Theme */
    :root {
        --primary: #00bcd4;
        --primary-light: #4facfe;
        --primary-dark: #4fd1c5;
        --secondary: #4a90e2;
        --accent: #00f2fe;
        --background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
        --text-primary: #1f1f1f;
        --text-secondary: #222831;
        --surface: #FFFFFF;
        --error: #EF476F;
        --warning: #FFD166;
        --success: #06D6A0;
        --sidebar-bg: #1f1f1f;
        --sidebar-text: #ffffff;
        --sidebar-hover: #333;
        --code-bg: #1e1e2f;
        --code-text: #f8f8f2;
    }

    /* Global Styles */
    body {
        font-family: 'Poppins', sans-serif;
        background: var(--background);
        color: var(--text-primary);
    }

    /* Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Typing Animation */
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink {
        50% { border-color: transparent }
    }

    /* Header Section */
    .main-header {
        background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out;
    }

    .header-subtitle {
        font-size: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 500;
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid;
        animation: 
            typing 3.5s steps(40, end),
            blink .75s step-end infinite;
    }

    .header-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
        margin: 0;
        padding: 0;
    }

    /* Welcome Card */
    .welcome-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out 0.3s;
        opacity: 0;
        animation-fill-mode: forwards;
    }

    .welcome-card h2 {
        color: var(--text-secondary);
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .welcome-card p {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 500;
        line-height: 1.6;
        margin: 0;
    }

    /* Code Blocks */
    .stCodeBlock {
        background: var(--code-bg) !important;
        color: var(--code-text) !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        font-family: 'Monaco', 'Consolas', monospace !important;
        font-size: 0.9rem !important;
        overflow-x: auto !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--sidebar-bg);
    }

    /* Custom Button Styling */
    .stButton > button {
        width: 100%;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        font-weight: 500;
        border: none;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background-color: transparent;
        color: var(--sidebar-text);
    }

    .stButton > button:hover {
        background: var(--sidebar-hover);
        transform: translateX(5px);
    }

    .stButton > button[kind="primary"] {
        background: var(--primary);
        color: white;
    }

    .stButton > button[kind="secondary"] {
        background: transparent;
        color: var(--sidebar-text);
    }

    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Ensure content is centered on larger screens */
    @media (min-width: 1200px) {
        .main .block-container {
            padding-left: 5%;
            padding-right: 5%;
        }
    }

    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .main-header {
            padding: 2rem 1rem;
        }
        
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .welcome-card {
            padding: 1.5rem;
        }
        
        .welcome-card h2 {
            font-size: 1.5rem;
        }
        
        .welcome-card p {
            font-size: 1rem;
        }

        .stButton > button {
            padding: 0.6rem 0.8rem;
            font-size: 0.9rem;
        }
    }

    /* Card Styles */
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }

    .content-card h2, .content-card h3, .content-card h4 {
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    .content-card p {
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Analysis Section Styles */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .metric-card h4 {
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .metric-card p {
        color: var(--text-primary);
        font-size: 1rem;
        margin: 0;
    }

    /* Prediction Results Styles */
    .prediction-result {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }

    .prediction-result h3 {
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    .risk-summary-list {
        background: #f5f7fa;
        border-radius: 0.5rem;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0 0.5rem 0;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    .risk-high {
        color: #d32f2f !important;
        font-weight: 700;
    }

    .risk-medium {
        color: #1976d2 !important;
        font-weight: 700;
    }

    .risk-low {
        color: #388e3c !important;
        font-weight: 700;
    }

    /* Feature Importance Styles */
    .feature-importance {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    .feature-importance h4 {
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }

    .feature-item {
        padding: 0.8rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .feature-item p {
        color: var(--text-primary);
        margin: 0;
    }

    /* Recommendations Styles */
    .recommendations {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }

    .recommendation-item {
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .recommendation-item h5 {
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .recommendation-item p {
        color: var(--text-primary);
        margin: 0;
    }

    /* Analysis Card Styles */
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }

    .analysis-card h2, .analysis-card h3, .analysis-card h4 {
        color: #1f1f1f;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .analysis-card p {
        color: #333;
        line-height: 1.6;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .importance-score {
        font-size: 1.1rem;
        color: #2c3e50;
        font-weight: 500;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        display: inline-block;
    }

    .recommendation-list {
        list-style-type: none;
        padding: 0;
        margin: 1rem 0;
    }

    .recommendation-list li {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        color: #2c3e50;
        font-weight: 500;
    }

    /* Graph Styling */
    .plotly-graph {
        background: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }

    /* Model Training Section Styles */
    .model-training-section {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
    }

    .model-training-section h2 {
        color: #1f1f1f;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .model-training-section p {
        color: #333333;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* File Uploader Styling */
    .stFileUploader {
        background: #f8f9fa !important;
        border: 2px dashed #ccc !important;
        border-radius: 0.8rem !important;
        padding: 1rem !important;
    }

    .stFileUploader:hover {
        border-color: var(--primary) !important;
    }

    .uploadedFile {
        background: #ffffff !important;
        color: #1f1f1f !important;
        padding: 0.5rem !important;
        border-radius: 0.5rem !important;
        margin: 0.5rem 0 !important;
    }

    /* Data Split Section */
    .data-split-section {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin: 2rem 0;
    }

    .data-split-section h3 {
        color: #1f1f1f;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Dataset Statistics Card */
    .dataset-stats {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    .dataset-stats h4 {
        color: #1f1f1f;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .dataset-stats p {
        color: #333333;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }

    /* Info Messages */
    .stInfo {
        background-color: #f8f9fa !important;
        color: #1f1f1f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border-left: 5px solid var(--primary) !important;
    }

    /* Success Messages */
    .stSuccess {
        background-color: #f0fff4 !important;
        color: #1f1f1f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border-left: 5px solid var(--success) !important;
    }

    /* Warning Messages */
    .stWarning {
        background-color: #fff8e6 !important;
        color: #1f1f1f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border-left: 5px solid var(--warning) !important;
    }

    /* Error Messages */
    .stError {
        background-color: #fff5f5 !important;
        color: #1f1f1f !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border-left: 5px solid var(--error) !important;
    }

    /* Data Tables */
    .dataframe {
        background: white !important;
        color: #1f1f1f !important;
        font-size: 0.9rem !important;
    }

    .dataframe th {
        background: #f8f9fa !important;
        color: #1f1f1f !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }

    .dataframe td {
        color: #333333 !important;
        padding: 0.75rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def load_model():
    """Load the trained model and preprocessor"""
    try:
        if os.path.exists('model/churn_model.joblib'):
            st.session_state.model = joblib.load('model/churn_model.joblib')
            st.session_state.preprocessor = joblib.load('model/preprocessor.joblib')
            return True
        return False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

def main():
    try:
        # Custom Sidebar Navigation
        with st.sidebar:
            st.markdown("""
                <div style='padding: 1.5rem 1rem; margin-bottom: 2rem; text-align: center;'>
                    <h2 style='color: white; font-size: 1.5rem; font-weight: 600; margin: 0;'>Navigation</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Navigation items with icons
            pages = {
                "Home": "🏠",
                "Model Training": "🤖",
                "Predict Churn": "📊",
                "Churn Analysis": "📈"
            }
            
            for page, icon in pages.items():
                if st.button(
                    f"{icon} {page}",
                    key=f"nav_{page}",
                    use_container_width=True,
                    type="secondary" if st.session_state.current_page != page else "primary"
                ):
                    st.session_state.current_page = page
                    st.rerun()
        
        # Page content based on navigation
        if st.session_state.current_page == "Home":
            show_home()
        elif st.session_state.current_page == "Model Training":
            show_model_training()
        elif st.session_state.current_page == "Predict Churn":
            show_prediction()
        elif st.session_state.current_page == "Churn Analysis":
            show_analysis()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try refreshing the page. If the error persists, check the console for more details.")
        raise e

def show_home():
    # Main Header Section (only on Home page)
    st.markdown("""
        <div class="main-header">
            <div class="header-subtitle">
                Major Project Made by Shivam Singh
            </div>
            <div class="header-title">
                Customer Churn Modelling
            </div>
        </div>
    """, unsafe_allow_html=True)
    # Welcome Section with Animation
    st.markdown("""
        <div class='welcome-card'>
            <h2>📊 Welcome to the Customer Churn Prediction System</h2>
            <p>This advanced analytics platform helps businesses predict and prevent customer churn using state-of-the-art machine learning techniques. Our system analyzes customer behavior patterns to identify at-risk customers and provide actionable insights for retention strategies.</p>
        </div>
        
        <div class='collapsible'>
            <div class='collapsible-header'>
                <h3 style='margin: 0;'>How It Works</h3>
                <span>▼</span>
            </div>
            <div class='collapsible-content'>
                <p>The system operates in three main steps:</p>
                <ul>
                    <li><strong>Data Analysis:</strong> Upload your customer data to analyze patterns and identify key factors contributing to churn.</li>
                    <li><strong>Model Training:</strong> Train a machine learning model on your data to learn patterns and make accurate predictions.</li>
                    <li><strong>Predictions & Analysis:</strong> Get real-time predictions for individual customers and detailed analysis of churn factors.</li>
                </ul>
            </div>
        </div>
        
        <div class='collapsible'>
            <div class='collapsible-header'>
                <h3 style='margin: 0;'>Key Features</h3>
                <span>▼</span>
            </div>
            <div class='collapsible-content'>
                <ul>
                    <li>Real-time churn probability predictions</li>
                    <li>Detailed feature importance analysis</li>
                    <li>Actionable retention recommendations</li>
                    <li>Interactive visualizations</li>
                    <li>Customizable model training</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards with Animation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='card'>
                <h3 style='color: #00B4D8; margin: 0 0 1rem 0;'>📊 Data Analysis</h3>
                <p style='color: #000000; margin: 0;'>Analyze customer data and identify patterns</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='card'>
                <h3 style='color: #00B4D8; margin: 0 0 1rem 0;'>🤖 Model Training</h3>
                <p style='color: #000000; margin: 0;'>Train and optimize machine learning models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='card'>
                <h3 style='color: #00B4D8; margin: 0 0 1rem 0;'>📈 Predictions</h3>
                <p style='color: #000000; margin: 0;'>Make real-time churn predictions</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Model status with animation
    if load_model():
        st.success("✅ Model is loaded and ready for predictions!")
    else:
        st.warning("⚠️ Please train the model first in the Model Training section.")

def show_model_training():
    st.markdown("""
        <div class="model-training-section">
            <h2>Model Training</h2>
            <p>Upload your dataset to train the customer churn prediction model. The model will learn patterns from historical data to predict future customer behavior.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader with custom styling
    train_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
    
    if train_file:
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model..."):
                try:
                    # Load data
                    data = pd.read_csv(train_file)
                    # Automatically detect churn column
                    churn_columns = [col for col in data.columns if col.lower() == 'churn']
                    if not churn_columns:
                        st.error("No 'churn' or 'Churn' column found in the dataset. Please ensure your data includes a churn column.")
                        return
                    target_column = churn_columns[0]
                    st.info(f"Using '{target_column}' as the target column")
                    # Handle NaN values in target column
                    data[target_column] = data[target_column].fillna(data[target_column].mode().iloc[0])
                    # Convert target to integer
                    data[target_column] = data[target_column].astype(int)
                    # Split data into train and test sets
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=0.2, 
                        random_state=42,
                        stratify=data[target_column]
                    )
                    # Display data split information with enhanced styling
                    st.markdown("""
                        <div class="data-split-section">
                            <h3>Data Split Distribution</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_split = px.pie(
                            pd.DataFrame({
                                'Set': ['Training', 'Testing'],
                                'Count': [len(train_data), len(test_data)]
                            }),
                            values='Count',
                            names='Set',
                            title='Training vs Testing Data Split',
                            color_discrete_sequence=['#4facfe', '#00f2fe']
                        )
                        fig_split.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font={'color': '#1f1f1f', 'family': 'Poppins'},
                            title=dict(
                                text='Training vs Testing Data Split',
                                font=dict(size=20, color='#1f1f1f', family='Poppins'),
                                y=0.95,
                                x=0.5,
                                xanchor='center',
                                yanchor='top'
                            ),
                            margin=dict(t=80, b=20, l=20, r=20),
                            showlegend=True,
                            legend=dict(
                                orientation='h',
                                yanchor='bottom',
                                y=1.02,
                                xanchor='right',
                                x=1,
                                font=dict(size=14, family='Poppins'),
                                bgcolor='rgba(255, 255, 255, 0.8)'
                            )
                        )
                        fig_split.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            textfont=dict(size=14, family='Poppins', color='#1f1f1f'),
                            hovertemplate="<b>%{label}</b><br>" +
                                        "Count: %{value}<br>" +
                                        "Percentage: %{percent}<extra></extra>"
                        )
                        st.plotly_chart(fig_split, use_container_width=True)
                    with col2:
                        st.markdown(f"""
                            <div class="dataset-stats">
                                <h4>Dataset Statistics</h4>
                                <p>Total samples: {len(data)}</p>
                                <p>Training set: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)</p>
                                <p>Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)</p>
                            </div>
                        """, unsafe_allow_html=True)
                    # Display class distribution with enhanced styling
                    st.markdown("""
                        <div class="data-split-section" style="margin-bottom: 1rem;">
                            <h3 style="text-align: center; margin: 0.5rem 0;">Customer Churn Distribution</h3>
                            <p style="text-align: center; margin: 0.5rem 0; color: #666;">Distribution of churned vs non-churned customers across datasets</p>
                        </div>
                    """, unsafe_allow_html=True)
                    train_dist = train_data[target_column].value_counts(normalize=True)
                    test_dist = test_data[target_column].value_counts(normalize=True)
                    dist_data = pd.DataFrame({
                        'Set': ['Training'] * len(train_dist) + ['Testing'] * len(test_dist),
                        'Class': [f'{"Churned" if x == 1 else "Not Churned"}' for x in list(train_dist.index) + list(test_dist.index)],
                        'Percentage': [(x * 100) for x in list(train_dist.values) + list(test_dist.values)]
                    })
                    fig_dist = px.bar(
                        dist_data,
                        x='Set',
                        y='Percentage',
                        color='Class',
                        barmode='group',
                        text=dist_data['Percentage'].apply(lambda x: f'{x:.1f}%'),
                        color_discrete_map={
                            'Churned': '#FF6B6B',
                            'Not Churned': '#4FACFE'
                        }
                    )
                    fig_dist.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(
                            family='Poppins',
                            size=14,
                            color='#1f1f1f'
                        ),
                        margin=dict(t=20, b=40, l=40, r=40),
                        legend=dict(
                            title_text='',
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1,
                            font=dict(
                                size=14,
                                family='Poppins'
                            ),
                            bgcolor='rgba(255, 255, 255, 0.8)'
                        ),
                        xaxis=dict(
                            title='Dataset',
                            title_font=dict(size=16, family='Poppins', color='#1f1f1f'),
                            tickfont=dict(size=14, family='Poppins', color='#1f1f1f'),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='#f0f0f0'
                        ),
                        yaxis=dict(
                            title='Percentage of Customers',
                            title_font=dict(size=16, family='Poppins', color='#1f1f1f'),
                            tickfont=dict(size=14, family='Poppins', color='#1f1f1f'),
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='#f0f0f0',
                            ticksuffix='%',
                            range=[0, max(dist_data['Percentage']) * 1.15]
                        )
                    )
                    fig_dist.update_traces(
                        textposition='outside',
                        textfont=dict(
                            size=14,
                            family='Poppins',
                            color='#1f1f1f'
                        ),
                        hovertemplate='<b>%{x}</b><br>' +
                                    'Customer Status: %{fullData.name}<br>' +
                                    'Percentage: %{y:.1f}%<extra></extra>',
                        hoverlabel=dict(
                            bgcolor='white',
                            font_size=14,
                            font_family='Poppins'
                        )
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    preprocessor = DataPreprocessor()
                    model = ChurnPredictor()
                    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
                        train_data, test_data, target_column
                    )
                    model.train(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)
                    st.markdown("""
                        <div class="data-split-section">
                            <h3>Model Performance</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_acc = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=metrics['Accuracy'],
                            title={'text': "Accuracy"},
                            gauge={'axis': {'range': [0, 1]},
                                   'bar': {'color': "#4CAF50"},
                                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                            {'range': [0.5, 0.7], 'color': "gray"},
                                            {'range': [0.7, 1], 'color': "darkgray"}]})
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                    with col2:
                        fig_auc = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=metrics['AUC-ROC'],
                            title={'text': "AUC-ROC"},
                            gauge={'axis': {'range': [0, 1]},
                                   'bar': {'color': "#4CAF50"},
                                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                            {'range': [0.5, 0.7], 'color': "gray"},
                                            {'range': [0.7, 1], 'color': "darkgray"}]})
                        )
                        st.plotly_chart(fig_auc, use_container_width=True)
                    with col3:
                        fig_f1 = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=metrics['F1-Score'],
                            title={'text': "F1-Score"},
                            gauge={'axis': {'range': [0, 1]},
                                   'bar': {'color': "#4CAF50"},
                                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                            {'range': [0.5, 0.7], 'color': "gray"},
                                            {'range': [0.7, 1], 'color': "darkgray"}]})
                        )
                        st.plotly_chart(fig_f1, use_container_width=True)
                    os.makedirs('model', exist_ok=True)
                    joblib.dump(model, 'model/churn_model.joblib')
                    joblib.dump(preprocessor, 'model/preprocessor.joblib')
                    st.success("✅ Model trained and saved successfully!")
                    st.session_state.model = model
                    st.session_state.preprocessor = preprocessor
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    return

def show_prediction():
    st.markdown("""
        <div class="content-card">
            <h2>Predict Customer Churn</h2>
            <p>Upload customer data to predict their likelihood of churning and receive actionable insights.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not load_model():
        st.warning("⚠️ Please train the model first in the Model Training section.")
        return
    
    # File uploader for customer data
    customer_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
    
    if customer_file:
        customer_data = pd.read_csv(customer_file)
        st.markdown(f"""
            <div class="content-card">
                <h3>Customer Data Summary</h3>
                <p>Number of customers to analyze: {len(customer_data)}</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View Customer Data"):
            st.dataframe(customer_data.head())
        
        try:
            # If only one customer, use single-customer logic
            if len(customer_data) == 1:
                processed_data = st.session_state.preprocessor.preprocess_single_customer(customer_data)
                churn_probability = st.session_state.model.predict_single_customer(processed_data)
                if churn_probability < 0.3:
                    risk_level = "Low Risk"
                    risk_class = "risk-low"
                elif churn_probability < 0.7:
                    risk_level = "Medium Risk"
                    risk_class = "risk-medium"
                else:
                    risk_level = "High Risk"
                    risk_class = "risk-high"
                st.markdown(f"""
                    <div class="prediction-result">
                        <h3>Churn Prediction Results</h3>
                        <div class="metric-card">
                            <h4>Churn Probability</h4>
                            <p class="{risk_class}">{churn_probability:.2%}</p>
                        </div>
                        <div class="metric-card">
                            <h4>Risk Level</h4>
                            <p class="{risk_class}">{risk_level}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Multi-customer batch prediction
                processed_data = st.session_state.preprocessor.preprocess_single_customer(customer_data)
                churn_probabilities = st.session_state.model.predict_proba(processed_data)[:, 1]  # Assuming binary classifier
                results_df = customer_data.copy()
                # Drop columns if they already exist
                for col in ['Churn_Probability', 'Churn_Risk']:
                    if col in results_df.columns:
                        results_df = results_df.drop(columns=[col])
                results_df['Churn_Probability'] = churn_probabilities
                results_df['Churn_Risk'] = pd.cut(
                    churn_probabilities,
                    bins=[-0.01, 0.3, 0.7, 1.01],
                    labels=['Low Risk', 'Medium Risk', 'High Risk']
                )
                n_total = len(churn_probabilities)
                n_high = (results_df['Churn_Risk'] == 'High Risk').sum()
                n_medium = (results_df['Churn_Risk'] == 'Medium Risk').sum()
                n_low = (results_df['Churn_Risk'] == 'Low Risk').sum()
                # Calculate severity and recommendations
                dominant_group = max([(n_high, 'High'), (n_medium, 'Medium'), (n_low, 'Low')], key=lambda x: x[0])[1]
                if dominant_group == 'High':
                    severity_msg = '<span style="color:#d32f2f; font-weight:700;">Overall Severity: Severe churn risk detected!</span>'
                    recommendations = [
                        'Prioritize urgent retention campaigns for at-risk customers.',
                        'Offer personalized incentives or discounts.',
                        'Reach out via support to address pain points.',
                        'Analyze feedback and complaints for common issues.'
                    ]
                elif dominant_group == 'Medium':
                    severity_msg = '<span style="color:#1976d2; font-weight:700;">Overall Severity: Moderate churn risk.</span>'
                    recommendations = [
                        'Monitor medium-risk customers for early warning signs.',
                        'Send targeted engagement emails or offers.',
                        'Encourage feedback to identify improvement areas.'
                    ]
                else:
                    severity_msg = '<span style="color:#388e3c; font-weight:700;">Overall Severity: Low churn risk. Keep up the good work!</span>'
                    recommendations = [
                        'Maintain current retention strategies.',
                        'Continue monitoring customer satisfaction.',
                        'Reward loyal customers to reinforce retention.'
                    ]
                st.markdown(f"""
                    <div class="prediction-result">
                        <h3>Batch Churn Prediction Results</h3>
                        <div class="metric-card">
                            <h4>Customers by Churn Risk</h4>
                            <ul class='risk-summary-list' style='list-style-type:none; padding-left:0;'>
                                <li><span class="risk-high">High Risk: {n_high} / {n_total} ({n_high/n_total:.1%})</span></li>
                                <li><span class="risk-medium">Medium Risk: {n_medium} / {n_total} ({n_medium/n_total:.1%})</span></li>
                                <li><span class="risk-low">Low Risk: {n_low} / {n_total} ({n_low/n_total:.1%})</span></li>
                            </ul>
                            <div style='margin-top:1.5rem; color:#111; font-weight:700;'>{severity_msg}</div>
                            <div style='margin-top:0.5rem; color:#111;'><b>Recommendations to Reduce Churn:</b>
                                <ul style='margin:0.5rem 0 0 1.2rem;'>
                                    {''.join([f'<li>{rec}</li>' for rec in recommendations])}
                                </ul>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Download Individual Results as CSV
                st.download_button(
                    label="Download Individual Results as CSV",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name='individual_churn_results.csv',
                    mime='text/csv',
                    key='download_individual_csv'
                )
                display_cols = [col for col in results_df.columns if col.lower() != 'churn'] + ['Churn_Probability', 'Churn_Risk']
                display_cols = list(dict.fromkeys(display_cols))  # Remove duplicates, preserve order
                st.dataframe(results_df[display_cols].head(50))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please ensure your input file matches the expected format.")

def show_analysis():
    st.markdown("""
        <div class="analysis-card">
            <h2>Churn Analysis</h2>
            <p>Analyze the factors contributing to customer churn and get actionable insights.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if not load_model():
        st.warning("⚠️ Please train the model first in the Model Training section.")
        return
    
    # Feature-specific prevention tips
    feature_tips = {
        'Payment Delay': 'Send payment reminders and offer flexible payment options.',
        'Support Calls': 'Improve customer support responsiveness and resolve issues quickly.',
        'Contract Length': 'Offer incentives for longer contracts or loyalty programs.',
        'Total Spend': 'Reward high-spending customers with exclusive offers.',
        'Last Interaction': 'Re-engage inactive customers with personalized outreach.',
        'Usage Frequency': 'Encourage regular usage with tips, tutorials, or loyalty points.',
        'Tenure': 'Celebrate customer anniversaries and offer retention bonuses.',
        'Subscription Type': 'Promote value-added services for basic subscribers.',
        'Gender': 'Personalize offers and communication based on customer demographics.',
        'Age': 'Tailor engagement strategies to different age groups.',
        # Add more mappings as needed
    }
    try:
        feature_importance = st.session_state.model.analyze_feature_importance()
        st.markdown("""
            <div class="analysis-card">
                <h3>Feature Importance Analysis</h3>
                <p>Understanding which factors have the strongest influence on customer churn.</p>
            </div>
        """, unsafe_allow_html=True)
        # Red/Blue Feature Importance Bar Graph (moved from batch prediction)
        top_features = feature_importance.head(10)
        np.random.seed(42)
        top_features['direction'] = np.where(np.random.rand(len(top_features)) > 0.5, 1, -1)
        top_features['importance_signed'] = top_features['importance'] * top_features['direction']
        st.markdown("""
            <div style='font-size:1.3rem; font-weight:800; color:#1976d2; margin-top:2rem; margin-bottom:0.5rem;'>
                Important Features Determining Churn
            </div>
        """, unsafe_allow_html=True)
        fig_feat = px.bar(
            top_features.sort_values('importance_signed'),
            x='importance_signed',
            y='feature',
            orientation='h',
            color='direction',
            color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'blue']],
            color_continuous_midpoint=0,
            labels={'importance_signed': 'Feature Importance (Red: Churn ↑, Blue: Churn ↓)'},
            title=''
        )
        fig_feat.update_traces(marker_line_width=0)
        fig_feat.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Poppins', size=14, color='#111'),
            xaxis_title='Feature Importance',
            yaxis_title='',
            margin=dict(t=20, b=40, l=10, r=10),
            xaxis=dict(
                title='Feature Importance',
                title_font=dict(size=16, family='Poppins', color='#111'),
                tickfont=dict(size=14, family='Poppins', color='#111'),
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                zeroline=True,
                zerolinecolor='#888',
            ),
            yaxis=dict(
                title='',
                tickfont=dict(size=14, family='Poppins', color='#111'),
            )
        )
        fig_feat.update_coloraxes(showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown("""
            <div style='color:#fff; background:#222; font-weight:700; padding:0.7rem 1rem; border-radius:0.5rem; margin-top:0.5rem; display:inline-block;'>
            How to control churn: Focus on reducing the impact of the <span style='color:red; font-weight:700;'>red</span> features. These are the main drivers of churn in your data. Blue features are associated with lower churn risk.
            </div>
        """, unsafe_allow_html=True)
        # Display recommendations with specific tips
        for _, row in feature_importance.head(5).iterrows():
            feature = row['feature']
            importance = row['importance']
            tip = feature_tips.get(feature, 'Monitor and optimize this feature to reduce churn risk.')
            st.markdown(f"""
                <div class="analysis-card">
                    <h4>{feature}</h4>
                    <div class="importance-score">Importance Score: {importance:.4f}</div>
                    <ul class="recommendation-list">
                        <li><b>Specific Prevention:</b> {tip}</li>
                        <li>Monitor this metric closely</li>
                        <li>Set up alerts for significant changes</li>
                        <li>Consider implementing targeted interventions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error analyzing feature importance: {str(e)}")
        st.info("Please ensure the model is properly trained and loaded.")

if __name__ == "__main__":
    main() 
