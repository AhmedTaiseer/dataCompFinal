import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# Load the trained model only
@st.cache_resource
def load_model():
    """Load only the trained model"""
    try:
        model = joblib.load('svm_classification_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, label_encoders
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None

# Load model
model, label_encoders = load_model()

if model is not None:
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Product Information")
        
        # Create input form
        with st.form("prediction_form"):
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                price = st.number_input(
                    "Price ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=50.0,
                    step=10.0
                )
                
                discount = st.number_input(
                    "Discount (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=5.0
                )
                
                stock = st.number_input(
                    "Stock Quantity",
                    min_value=0,
                    max_value=10000,
                    value=100,
                    step=10
                )
            
            with col1b:
                review_count = st.number_input(
                    "Review Count",
                    min_value=0,
                    max_value=100000,
                    value=50,
                    step=10
                )
                
                seller_rating = st.number_input(
                    "Seller Rating",
                    min_value=0.0,
                    max_value=5.0,
                    value=4.5,
                    step=0.1
                )
                
                shipping_time = st.number_input(
                    "Shipping Time (days)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    step=1
                )
            
            with col1c:
                if 'category' in label_encoders:
                    categories = label_encoders['category'].classes_.tolist()
                    category = st.selectbox("Category", options=categories)
                else:
                    category = "Electronics"
                
                if 'device' in label_encoders:
                    devices = label_encoders['device'].classes_.tolist()
                    device = st.selectbox("Device Used", options=devices)
                else:
                    device = "Mobile"
                
                if 'payment_method' in label_encoders:
                    payment_methods = label_encoders['payment_method'].classes_.tolist()
                    payment_method = st.selectbox("Payment Method", options=payment_methods)
                else:
                    payment_method = "Credit Card"
            
            if 'season' in label_encoders:
                seasons = label_encoders['season'].classes_.tolist()
                season = st.selectbox("Season", options=seasons)
            else:
                season = "Summer"
            
            submitted = st.form_submit_button("Predict Return Probability", use_container_width=True)
    
    with col2:
        st.markdown("### Model Information")
        st.markdown("""
        <div class="metric-card">
            <strong>Model Type:</strong> Linear SVM<br>
            <strong>Task:</strong> Binary Classification<br>
            <strong>Target:</strong> Product Return Prediction<br>
            <strong>Features:</strong> Price, Discount, Final Price,<br>
            Reviews, Stock, Seller Rating, Shipping Time,<br>
            Category, Device, Payment Method, Season
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
    
    if submitted:
        with st.spinner("Analyzing product data..."):
            try:
                # Calculate final price
                final_price = price * (1 - discount / 100)
                
                # Create feature array in the exact order the model expects
                # First, encode categorical variables
                cat_encoded = []
                
                if 'category' in label_encoders:
                    try:
                        cat_encoded.append(label_encoders['category'].transform([category])[0])
                    except:
                        cat_encoded.append(label_encoders['category'].transform([label_encoders['category'].classes_[0]])[0])
                else:
                    cat_encoded.append(0)
                
                if 'device' in label_encoders:
                    try:
                        cat_encoded.append(label_encoders['device'].transform([device])[0])
                    except:
                        cat_encoded.append(label_encoders['device'].transform([label_encoders['device'].classes_[0]])[0])
                else:
                    cat_encoded.append(0)
                
                if 'payment_method' in label_encoders:
                    try:
                        cat_encoded.append(label_encoders['payment_method'].transform([payment_method])[0])
                    except:
                        cat_encoded.append(label_encoders['payment_method'].transform([label_encoders['payment_method'].classes_[0]])[0])
                else:
                    cat_encoded.append(0)
                
                if 'season' in label_encoders:
                    try:
                        cat_encoded.append(label_encoders['season'].transform([season])[0])
                    except:
                        cat_encoded.append(0)
                else:
                    cat_encoded.append(0)
                
                # Create numeric feature array
                # Order: price, discount, final_price, review_count, stock, seller_rating, shipping_time_days
                numeric_features = np.array([[
                    price,
                    discount,
                    final_price,
                    review_count,
                    stock,
                    seller_rating,
                    shipping_time
                ]])
                
                # Combine numeric and categorical features
                categorical_features = np.array([cat_encoded])
                final_features = np.hstack([numeric_features, categorical_features])
                
                # Make prediction
                prediction_proba = model.predict_proba(final_features)[0]
                prediction = model.predict(final_features)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("## Prediction Results")
                
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
                    st.metric(label="Prediction", value=prediction_text)
                
                with res_col3:
                    confidence = max(prediction_proba) * 100
                    st.metric(label="Confidence", value=f"{confidence:.1f}%")
                
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
                    
                    if price > 100:
                        st.markdown("- [!] High price may increase return risk")
                    elif price < 20:
                        st.markdown("- [OK] Low price generally reduces return risk")
                    else:
                        st.markdown("- [v] Price is in optimal range")
                    
                    if discount > 30:
                        st.markdown("- [!] High discount might indicate quality issues")
                    elif discount > 0:
                        st.markdown("- [v] Moderate discount is attractive")
                    else:
                        st.markdown("- [i] No discount offered")
                    
                    if shipping_time > 7:
                        st.markdown("- [!] Long shipping time increases return likelihood")
                    elif shipping_time <= 3:
                        st.markdown("- [OK] Fast shipping reduces return risk")
                    else:
                        st.markdown("- [v] Standard shipping time")
                
                with col_b:
                    st.markdown("**Quality Indicators:**")
                    
                    if seller_rating < 3.5:
                        st.markdown("- [!] Low seller rating increases return risk")
                    elif seller_rating >= 4.5:
                        st.markdown("- [OK] High seller rating reduces returns")
                    else:
                        st.markdown("- [v] Acceptable seller rating")
                    
                    if review_count < 10:
                        st.markdown("- [!] Few reviews may indicate new product")
                    elif review_count > 100:
                        st.markdown("- [OK] Many reviews suggest established product")
                    else:
                        st.markdown("- [v] Adequate number of reviews")
                    
                    if stock > 1000:
                        st.markdown("- [i] High stock might indicate overstocking")
                    else:
                        st.markdown("- [v] Reasonable stock level")
                
                # Actionable insights
                st.markdown("### Actionable Insights")
                
                if probability > 50:
                    if discount > 20:
                        st.markdown("- Consider reducing discount and improving product quality perception")
                    if seller_rating < 4.0:
                        st.markdown("- Work on improving seller rating through better customer service")
                    if shipping_time > 5:
                        st.markdown("- Optimize shipping processes or offer expedited shipping options")
                    if review_count < 20:
                        st.markdown("- Encourage more customer reviews to build trust")
                else:
                    st.markdown("- Current configuration is promising for low returns")
                    if discount < 20:
                        st.markdown("- Consider small discounts to boost sales without increasing return risk")
                    st.markdown("- Maintain high seller rating and fast shipping")
                
                st.markdown("---")
                st.caption("Disclaimer: This prediction is based on machine learning model analysis and should be used as a guideline, not an absolute guarantee.")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please try again with different input values.")

else:
    st.warning("""
    ### Getting Started
    
    To use this app, you need:
    
    1. Trained model file: `svm_classification_model.pkl`
    2. Label encoders file: `label_encoders.pkl`
    
    Run your training script to generate these files, then restart the app.
    """)

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;">Built with Streamlit | Amazon Return Prediction Model</div>', unsafe_allow_html=True)
