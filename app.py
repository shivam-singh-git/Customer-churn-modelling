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

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

def load_model():
    """Load the trained model and preprocessor"""
    if os.path.exists('model/churn_model.joblib'):
        st.session_state.model = joblib.load('model/churn_model.joblib')
        st.session_state.preprocessor = joblib.load('model/preprocessor.joblib')
        return True
    return False

def main():
    st.title("Customer Churn Prediction Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Model Training", "Predict Churn", "Churn Analysis"]
    )
    
    if page == "Home":
        show_home()
    elif page == "Model Training":
        show_model_training()
    elif page == "Predict Churn":
        show_prediction()
    elif page == "Churn Analysis":
        show_analysis()

def show_home():
    st.header("Welcome to Customer Churn Prediction System")
    st.write("""
    This application helps you:
    1. Train a machine learning model to predict customer churn
    2. Make real-time predictions for individual customers
    3. Analyze factors contributing to customer churn
    4. Get actionable insights to improve customer retention
    """)
    
    # Display model status
    if load_model():
        st.success("Model is loaded and ready for predictions!")
    else:
        st.warning("Please train the model first in the Model Training section.")

def show_model_training():
    st.header("Model Training")
    
    # File uploader for training data
    train_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
    
    if train_file:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Initialize preprocessor and model
                preprocessor = DataPreprocessor()
                model = ChurnPredictor()
                
                # Load data
                data = pd.read_csv(train_file)
                
                # Automatically detect churn column
                churn_columns = [col for col in data.columns if col.lower() == 'churn']
                if not churn_columns:
                    st.error("No 'churn' or 'Churn' column found in the dataset. Please ensure your data includes a churn column.")
                    return
                
                target_column = churn_columns[0]  # Use the first matching column
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
                    stratify=data[target_column]  # Ensure balanced split
                )
                
                # Display split information with pie chart
                st.subheader("Data Split Distribution")
                split_data = pd.DataFrame({
                    'Set': ['Training', 'Testing'],
                    'Count': [len(train_data), len(test_data)]
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_split = px.pie(split_data, values='Count', names='Set', 
                                     title='Training vs Testing Data Split')
                    st.plotly_chart(fig_split, use_container_width=True)
                with col2:
                    st.write("Dataset Statistics:")
                    st.write(f"Total samples: {len(data)}")
                    st.write(f"Training set: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
                    st.write(f"Test set: {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
                
                # Display class distribution with bar chart
                st.subheader("Class Distribution")
                train_dist = train_data[target_column].value_counts(normalize=True)
                test_dist = test_data[target_column].value_counts(normalize=True)
                
                # Create comparison bar chart
                dist_data = pd.DataFrame({
                    'Set': ['Training'] * len(train_dist) + ['Testing'] * len(test_dist),
                    'Class': list(train_dist.index) + list(test_dist.index),
                    'Percentage': list(train_dist.values) + list(test_dist.values)
                })
                
                fig_dist = px.bar(dist_data, x='Set', y='Percentage', color='Class',
                                barmode='group', title='Churn Distribution in Training and Test Sets')
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Display feature distributions
                st.subheader("Feature Distributions")
                feature_cols = [col for col in data.columns if col not in [target_column, 'CustomerID']]
                
                # Create tabs for different types of features
                tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
                
                with tab1:
                    numerical_cols = data[feature_cols].select_dtypes(include=[np.number]).columns
                    for col in numerical_cols:
                        fig = px.histogram(data, x=col, color=target_column,
                                         title=f'Distribution of {col} by Churn Status',
                                         nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    categorical_cols = data[feature_cols].select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        fig = px.bar(data.groupby([col, target_column]).size().reset_index(name='count'),
                                   x=col, y='count', color=target_column,
                                   title=f'Distribution of {col} by Churn Status',
                                   barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Preprocess data
                X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
                    train_data, test_data, target_column
                )
                
                # Train model
                model.train(X_train, y_train)
                
                # Evaluate model
                metrics = model.evaluate(X_test, y_test)
                
                # Display metrics with gauge charts
                st.subheader("Model Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig_acc = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=metrics['Accuracy'],
                        title={'text': "Accuracy"},
                        gauge={'axis': {'range': [0, 1]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                        {'range': [0.5, 0.7], 'color': "gray"},
                                        {'range': [0.7, 1], 'color': "darkgray"}]}))
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    fig_auc = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=metrics['AUC-ROC'],
                        title={'text': "AUC-ROC"},
                        gauge={'axis': {'range': [0, 1]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                        {'range': [0.5, 0.7], 'color': "gray"},
                                        {'range': [0.7, 1], 'color': "darkgray"}]}))
                    st.plotly_chart(fig_auc, use_container_width=True)
                
                with col3:
                    fig_f1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=metrics['F1-Score'],
                        title={'text': "F1-Score"},
                        gauge={'axis': {'range': [0, 1]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                        {'range': [0.5, 0.7], 'color': "gray"},
                                        {'range': [0.7, 1], 'color': "darkgray"}]}))
                    st.plotly_chart(fig_f1, use_container_width=True)
                
                # Save model and preprocessor
                os.makedirs('model', exist_ok=True)
                joblib.dump(model, 'model/churn_model.joblib')
                joblib.dump(preprocessor, 'model/preprocessor.joblib')
                
                st.success("Model trained and saved successfully!")
                st.session_state.model = model
                st.session_state.preprocessor = preprocessor

def show_prediction():
    st.header("Predict Customer Churn")
    
    if not load_model():
        st.warning("Please train the model first in the Model Training section.")
        return
    
    st.write("""
    ### How to Use This Section
    1. Upload a CSV file containing customer data to predict their churn probability
    2. The file should contain the same features as the training data (except the Churn column)
    3. The model will analyze the data and provide:
       - Churn probability score
       - Risk level assessment
       - Key factors contributing to the prediction
       - Recommendations for retention
    """)
    
    # File uploader for customer data
    customer_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
    
    if customer_file:
        customer_data = pd.read_csv(customer_file)
        
        # Display uploaded data summary
        st.subheader("Customer Data Summary")
        st.write(f"Number of customers to analyze: {len(customer_data)}")
        
        # Show sample of the data
        with st.expander("View Customer Data"):
            st.dataframe(customer_data.head())
        
        # Preprocess customer data
        processed_data = st.session_state.preprocessor.preprocess_single_customer(customer_data)
        
        # Make prediction
        churn_probability = st.session_state.model.predict_single_customer(processed_data)
        
        # Determine risk level
        if churn_probability < 0.3:
            risk_level = "Low Risk"
            risk_color = "green"
        elif churn_probability < 0.7:
            risk_level = "Medium Risk"
            risk_color = "orange"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        # Display prediction results
        st.subheader("Churn Prediction Results")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.2%}",
                help="This percentage indicates how likely the customer is to churn. Higher percentage means higher risk of churn."
            )
        
        with col2:
            st.metric(
                "Risk Level",
                risk_level,
                help="Risk level categorization based on churn probability: Low (<30%), Medium (30-70%), High (>70%)"
            )
        
        with col3:
            st.metric(
                "Retention Priority",
                "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.3 else "Low",
                help="Recommended priority level for customer retention efforts"
            )
        
        # Get feature contributions
        feature_contributions = st.session_state.model.get_feature_contributions(processed_data)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': processed_data.columns,
            'contribution': feature_contributions[0]
        }).sort_values('contribution', key=abs, ascending=False)
        
        # Display feature contributions
        st.subheader("Key Factors Contributing to Prediction")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Bar Chart", "Detailed Analysis"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feature_importance['contribution'],
                y=feature_importance['feature'],
                orientation='h',
                marker_color=['red' if x > 0 else 'blue' for x in feature_importance['contribution']]
            ))
            fig.update_layout(
                title="Feature Contributions to Churn Prediction",
                xaxis_title="Contribution to Churn Probability",
                yaxis_title="Feature",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **How to interpret the chart:**
            - Red bars indicate features that increase churn probability
            - Blue bars indicate features that decrease churn probability
            - Longer bars indicate stronger influence on the prediction
            """)
        
        with tab2:
            # Display top contributing features
            st.write("Top Features Contributing to Churn Risk:")
            for _, row in feature_importance.head(5).iterrows():
                feature = row['feature']
                contribution = row['contribution']
                direction = "increases" if contribution > 0 else "decreases"
                st.write(f"- **{feature}**: {direction} churn probability by {abs(contribution):.2%}")
        
        # Generate recommendations
        st.subheader("Recommended Actions")
        
        # Get top risk factors
        risk_factors = feature_importance[feature_importance['contribution'] > 0].head(3)
        
        if not risk_factors.empty:
            st.write("**Primary Risk Factors:**")
            for _, row in risk_factors.iterrows():
                feature = row['feature']
                contribution = row['contribution']
                st.write(f"- **{feature}** (contribution: {contribution:.2%})")
                
                # Feature-specific recommendations
                if feature == 'Tenure':
                    st.write("  • Consider offering loyalty rewards")
                    st.write("  • Implement a customer appreciation program")
                elif feature == 'Support Calls':
                    st.write("  • Review support process efficiency")
                    st.write("  • Consider proactive support outreach")
                elif feature == 'Payment Delay':
                    st.write("  • Review payment terms")
                    st.write("  • Consider flexible payment options")
                else:
                    st.write("  • Monitor this metric closely")
                    st.write("  • Consider targeted interventions")
        
        # Add retention strategies
        st.write("**General Retention Strategies:**")
        st.write("""
        1. Proactive Engagement:
           - Regular check-ins
           - Personalized communications
           - Value-added services
        
        2. Customer Support:
           - Quick response times
           - Proactive issue resolution
           - Dedicated support channel
        
        3. Value Enhancement:
           - Loyalty programs
           - Special offers
           - Premium features
        """)

def show_analysis():
    st.header("Churn Analysis")
    
    if not load_model():
        st.warning("Please train the model first in the Model Training section.")
        return
    
    # Display feature importance
    st.subheader("Feature Importance Analysis")
    try:
        feature_importance = st.session_state.model.analyze_feature_importance()
        
        # Plot feature importance
        fig = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            title="Top 10 Most Important Features"
        )
        st.plotly_chart(fig)
        
        # Display recommendations
        st.subheader("Recommendations to Reduce Churn")
        top_features = feature_importance.head(5)
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            st.write(f"**{feature}** (Importance: {importance:.4f})")
            st.write("""
            - Monitor this metric closely
            - Set up alerts for significant changes
            - Consider implementing targeted interventions
            """)
    except Exception as e:
        st.error(f"Error analyzing feature importance: {str(e)}")
        st.info("Please ensure the model is properly trained and loaded.")

if __name__ == "__main__":
    main() 