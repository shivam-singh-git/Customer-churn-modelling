import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ChurnPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            objective='binary:logistic'
        )
        self.feature_importance = None
        self.shap_values = None
        self.feature_names = None
        
    def train(self, X_train, y_train):
        """Train the model"""
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Validate target variable
        unique_values = np.unique(y_train)
        if len(unique_values) != 2:
            raise ValueError(f"Target variable should have exactly 2 unique values. Found {len(unique_values)} values: {unique_values}")
        
        # Ensure target is integer type
        y_train = y_train.astype(int)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Ensure target is integer type
        y_test = y_test.astype(int)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def analyze_feature_importance(self):
        """Analyze feature importance using model's built-in feature importance"""
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
            
        # Get feature importance from the model
        importance = self.model.feature_importances_
        
        # Create DataFrame with feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance"""
        if self.feature_importance is None:
            self.analyze_feature_importance()
            
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.feature_importance.head(top_n),
                   x='importance', y='feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_shap_summary(self, X_train):
        """Plot SHAP summary plot"""
        if X_train is None:
            raise ValueError("Training data is required for SHAP analysis")
            
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X_train)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X_train, plot_type="bar")
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, model_path):
        """Save the trained model"""
        joblib.dump(self.model, model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        self.model = joblib.load(model_path)
    
    def predict_single_customer(self, customer_data):
        """Predict churn probability for a single customer"""
        prediction = self.model.predict_proba(customer_data)[:, 1]
        return prediction[0]
    
    def get_feature_contributions(self, customer_data):
        """Get SHAP values for a single customer"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(customer_data)
        return shap_values 