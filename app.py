import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Amazon Return Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #232F3E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-high-risk {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .prediction-low-risk {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    .stButton > button {
        background-color: #FF9900;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #FF8800;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">Amazon Product Return Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict whether a product is likely to be returned based on various features</div>', unsafe_allow_html=True)

# Load the trained model and preprocessors
@st.cache_resource
def load_models():
    """Load the trained model and preprocessing objects"""
    try:
        model = joblib.load('svm_classification_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        numeric_imputer = joblib.load('numeric_imputer.pkl')
        
        # Try to load categorical imputer if it exists
        try:
            categorical_imputer = joblib.load('categorical_imputer.pkl')
        except:
            categorical_imputer = None
            
        return model, label_encoders, numeric_imputer, categorical_imputer
    except FileNotFoundError:
        st.error("""
        Warning: Model files not found! Please ensure you have trained and saved the model first.
        
        Run your original training script to generate:
        - svm_classification_model.pkl
        - label_encoders.pkl
        - numeric_imputer.pkl
        - categorical_imputer.pkl (optional)
        """)
        return None, None, None, None

# Load models
model, label_encoders, numeric_imputer, categorical_imputer = load_models()

if model is not None:
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Product Information")
        
        # Create input form
        with st.form("prediction_form"):
            # Row 1: Product details
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                price = st.number_input(
                    "Price ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=50.0,
                    step=10.0,
                    help="Original price of the product"
                )
                
                discount = st.number_input(
                    "Discount (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=5.0,
                    help="Discount percentage applied"
                )
                
                stock = st.number_input(
                    "Stock Quantity",
                    min_value=0,
                    max_value=10000,
                    value=100,
                    step=10,
                    help="Number of units in stock"
                )
            
            with col1b:
                review_count = st.number_input(
                    "Review Count",
                    min_value=0,
                    max_value=100000,
                    value=50,
                    step=10,
                    help="Number of customer reviews"
                )
                
                seller_rating = st.number_input(
                    "Seller Rating",
                    min_value=0.0,
                    max_value=5.0,
                    value=4.5,
                    step=0.1,
                    help="Seller's average rating (0-5)"
                )
                
                shipping_time = st.number_input(
                    "Shipping Time (days)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1,
                    help="Estimated shipping time in days"
                )
            
            with col1c:
                # Categorical features (if they exist in the training data)
                if 'category' in label_encoders:
                    categories = label_encoders['category'].classes_.tolist()
                    category = st.selectbox(
                        "Category",
                        options=categories,
                        help="Product category"
                    )
                else:
                    category = st.text_input("Category", value="Electronics")
                    categories = []
                
                if 'device' in label_encoders:
                    devices = label_encoders['device'].classes_.tolist()
                    device = st.selectbox(
                        "Device Used",
                        options=devices,
                        help="Device used for purchase"
                    )
                else:
                    device = st.selectbox("Device Used", ["Mobile", "Desktop", "Tablet"])
                
                if 'payment_method' in label_encoders:
                    payment_methods = label_encoders['payment_method'].classes_.tolist()
                    payment_method = st.selectbox(
                        "Payment Method",
                        options=payment_methods,
                        help="Payment method used"
                    )
                else:
                    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal"])
            
            # Additional features if they exist
            if 'season' in label_encoders:
                seasons = label_encoders['season'].classes_.tolist()
                season = st.selectbox("Season", options=seasons)
            else:
                season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
            
            # Submit button
            submitted = st.form_submit_button("Predict Return Probability", use_container_width=True)
    
    with col2:
        st.markdown("### Model Information")
        st.markdown("""
        <div class="metric-card">
            <strong>Model Type:</strong> Linear SVM<br>
            <strong>Task:</strong> Binary Classification<br>
            <strong>Target:</strong> Product Return Prediction<br>
            <strong>Features Used:</strong> Price, Discount, Reviews,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;Stock, Seller Rating, Shipping Time,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;Category, Device, Payment Method
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Tips to Reduce Returns")
        st.info("""
        - Provide high-quality product images
        - Write detailed product descriptions
        - Encourage honest customer reviews
        - Offer faster shipping options
        - Optimize pricing and discounts
        """)
    
    # Make prediction when form is submitted
    if submitted:
        with st.spinner("Analyzing product data..."):
            # Prepare input data
            input_data = {}
            
            # Numeric features
            input_data['price'] = price
            input_data['discount'] = discount
            input_data['final_price'] = price * (1 - discount/100)
            input_data['review_count'] = review_count
            input_data['stock'] = stock
            input_data['seller_rating'] = seller_rating
            input_data['shipping_time_days'] = shipping_time
            
            # Categorical features - encode them
            if 'category' in label_encoders:
                try:
                    input_data['category'] = label_encoders['category'].transform([category])[0]
                except:
                    # If category not in training, use most common
                    input_data['category'] = label_encoders['category'].transform([label_encoders['category'].classes_[0]])[0]
            else:
                input_data['category'] = 0
            
            if 'device' in label_encoders:
                try:
                    input_data['device'] = label_encoders['device'].transform([device])[0]
                except:
                    input_data['device'] = label_encoders['device'].transform([label_encoders['device'].classes_[0]])[0]
            else:
                input_data['device'] = 0
            
            if 'payment_method' in label_encoders:
                try:
                    input_data['payment_method'] = label_encoders['payment_method'].transform([payment_method])[0]
                except:
                    input_data['payment_method'] = label_encoders['payment_method'].transform([label_encoders['payment_method'].classes_[0]])[0]
            else:
                input_data['payment_method'] = 0
            
            if 'season' in label_encoders:
                try:
                    input_data['season'] = label_encoders['season'].transform([season])[0]
                except:
                    input_data['season'] = 0
            else:
                input_data['season'] = 0
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all expected columns are present
            expected_columns = numeric_imputer.statistics_.shape[0] if hasattr(numeric_imputer, 'statistics_') else len(input_data)
            
            # Impute missing values if necessary
            numeric_cols = ['price', 'discount', 'final_price', 'review_count', 'stock', 
                          'seller_rating', 'shipping_time_days']
            input_numeric = input_df[numeric_cols]
            input_numeric_imputed = numeric_imputer.transform(input_numeric)
            
            # Combine features
            final_input = input_numeric_imputed
            
            # Add categorical columns if they exist in training
            categorical_cols = [col for col in ['category', 'device', 'payment_method', 'season'] 
                               if col in input_df.columns]
            if categorical_cols:
                input_categorical = input_df[categorical_cols].values
                final_input = np.hstack([final_input, input_categorical])
            
            # Make prediction
            prediction_proba = model.predict_proba(final_input)[0]
            prediction = model.predict(final_input)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## Prediction Results")
            
            # Create three columns for results
            res_col1, res_col2, res_col3 = st.columns(3)
            
            probability = prediction_proba[1] * 100
            
            with res_col1:
                st.metric(
                    label="Return Probability",
                    value=f"{probability:.1f}%",
                    delta="High Risk" if probability > 50 else "Low Risk"
                )
            
            with res_col2:
                prediction_text = "Likely to be Returned" if prediction == 1 else "Unlikely to be Returned"
                st.metric(
                    label="Prediction",
                    value=prediction_text
                )
            
            with res_col3:
                confidence = max(prediction_proba) * 100
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1f}%"
                )
            
            # Risk assessment box
            if probability > 70:
                st.markdown(f"""
                <div class="prediction-box prediction-high-risk">
                    <h2>HIGH RETURN RISK</h2>
                    <p>This product has a {probability:.1f}% probability of being returned.</p>
                    <p><strong>Recommendation:</strong> Review product quality, improve descriptions, or optimize pricing.</p>
                </div>
                """, unsafe_allow_html=True)
            elif probability > 40:
                st.markdown(f"""
                <div class="prediction-box" style="background-color: #fff3cd; border: 2px solid #ffc107;">
                    <h2>MODERATE RETURN RISK</h2>
                    <p>This product has a {probability:.1f}% probability of being returned.</p>
                    <p><strong>Recommendation:</strong> Monitor closely and consider slight improvements.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box prediction-low-risk">
                    <h2>LOW RETURN RISK</h2>
                    <p>This product has a {probability:.1f}% probability of being returned.</p>
                    <p><strong>Recommendation:</strong> Product seems promising for continued sales.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature impact analysis
            st.markdown("### Key Factors Affecting This Prediction")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Product Metrics:**")
                
                # Price impact
                if price > 100:
                    price_impact = "[!] High price may increase return risk"
                elif price < 20:
                    price_impact = "[OK] Low price generally reduces return risk"
                else:
                    price_impact = "[v] Price is in optimal range"
                st.markdown(f"- {price_impact}")
                
                # Discount impact
                if discount > 30:
                    discount_impact = "[!] High discount might indicate quality issues"
                elif discount > 0:
                    discount_impact = "[v] Moderate discount is attractive"
                else:
                    discount_impact = "[i] No discount offered"
                st.markdown(f"- {discount_impact}")
                
                # Shipping impact
                if shipping_time > 7:
                    shipping_impact = "[!] Long shipping time increases return likelihood"
                elif shipping_time <= 3:
                    shipping_impact = "[OK] Fast shipping reduces return risk"
                else:
                    shipping_impact = "[v] Standard shipping time"
                st.markdown(f"- {shipping_impact}")
            
            with col_b:
                st.markdown("**Quality Indicators:**")
                
                # Rating impact
                if seller_rating < 3.5:
                    rating_impact = "[!] Low seller rating increases return risk"
                elif seller_rating >= 4.5:
                    rating_impact = "[OK] High seller rating reduces returns"
                else:
                    rating_impact = "[v] Acceptable seller rating"
                st.markdown(f"- {rating_impact}")
                
                # Review count impact
                if review_count < 10:
                    review_impact = "[!] Few reviews may indicate new product"
                elif review_count > 100:
                    review_impact = "[OK] Many reviews suggest established product"
                else:
                    review_impact = "[v] Adequate number of reviews"
                st.markdown(f"- {review_impact}")
                
                # Stock impact
                if stock > 1000:
                    stock_impact = "[i] High stock might indicate overstocking"
                else:
                    stock_impact = "[v] Reasonable stock level"
                st.markdown(f"- {stock_impact}")
            
            # Actionable insights
            st.markdown("### Actionable Insights")
            
            insights = []
            if probability > 50:
                if discount > 20:
                    insights.append("- Consider reducing discount and improving product quality perception")
                if seller_rating < 4.0:
                    insights.append("- Work on improving seller rating through better customer service")
                if shipping_time > 5:
                    insights.append("- Optimize shipping processes or offer expedited shipping options")
                if review_count < 20:
                    insights.append("- Encourage more customer reviews to build trust")
            else:
                insights.append("- Current configuration is promising for low returns")
                if discount < 20:
                    insights.append("- Consider small discounts to boost sales without increasing return risk")
                insights.append("- Maintain high seller rating and fast shipping")
            
            for insight in insights:
                st.markdown(insight)
            
            # Disclaimer
            st.markdown("---")
            st.caption("Disclaimer: This prediction is based on machine learning model analysis and should be used as a guideline, not an absolute guarantee. Actual return behavior may vary based on multiple factors.")

