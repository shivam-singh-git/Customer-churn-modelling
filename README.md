# Customer Churn Prediction System

This is a comprehensive system for predicting customer churn using machine learning. The system includes data preprocessing, model training, prediction, and analysis capabilities, all wrapped in a user-friendly Streamlit interface.


Link to the deployed project via streamlit : https://shivam-singh-git-customer-churn-modelling-app-opvg2w.streamlit.app/

## Features

- Data preprocessing and feature engineering
- Machine learning model training using XGBoost
- Real-time churn prediction for individual customers
- Feature importance analysis using SHAP values
- Interactive visualizations and insights
- User-friendly web interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/shivam-singh-git/Customer-churn-modelling.git
cd customer-churn-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
customer-churn-prediction/
├── app.py                 # Main Streamlit application
├── data_preprocessing.py  # Data preprocessing module
├── model_training.py      # Model training and evaluation
├── requirements.txt       # Project dependencies
├── README.md             # This file
└── dataset/              # Directory for your dataset
    ├── training.csv      # Training data
    └── testing.csv       # Testing data
```

## Usage

1. Prepare your dataset:
   - Split your data into training and testing sets
   - Save them as CSV files in the `dataset` folder
   - Ensure your data includes both numerical and categorical features
   - The target column should be binary (0 for no churn, 1 for churn)

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Using the Application:
   - Navigate through different sections using the sidebar
   - Upload your training and testing data
   - Train the model and view performance metrics
   - Make predictions for new customers
   - Analyze feature importance and get recommendations

## Model Training

1. Go to the "Model Training" section
2. Upload your training and testing data files (CSV format)
3. Select the target column (churn indicator)
4. Click "Train Model" to start the training process
5. View the model performance metrics

## Making Predictions

1. Go to the "Predict Churn" section
2. Upload a customer data file (CSV format)
3. View the churn probability and feature contributions
4. Analyze which factors are most influencing the prediction

## Churn Analysis

1. Go to the "Churn Analysis" section
2. View feature importance plots
3. Get recommendations for reducing churn
4. Analyze patterns in customer behavior

## Deployment

To deploy the application:

1. Create a `requirements.txt` file (already included)
2. Choose a hosting platform (Streamlit Cloud, Heroku, etc.)
3. Follow the platform-specific deployment instructions

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
