import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Amazon Return Predictor",
    page_icon="📦",
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

# Load all necessary preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and all preprocessing objects"""
    try:
        model = joblib.load('svm_classification_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        
        # Try to load feature info if available
        try:
            feature_info = joblib.load('feature_info.pkl')
            feature_names = feature_info['all_features']
            numeric_features = feature_info['numeric_features']
            categorical_features = feature_info['categorical_features']
        except:
            # Fallback to default feature names
            feature_names = None
            numeric_features = ['price', 'discount', 'final_price', 'review_count', 'stock', 'seller_rating', 'shipping_time_days']
            categorical_features = ['category', 'device', 'payment_method', 'season']
        
        return model, label_encoders, numeric_features, categorical_features, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please run project.py first to generate the model files.")
        return None, None, None, None, None

# Load model and preprocessors
model, label_encoders, numeric_features, categorical_features, feature_names = load_model_and_preprocessors()

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
                # Category dropdown
                if 'category' in label_encoders:
                    categories = label_encoders['category'].classes_.tolist()
                    category = st.selectbox("Category", options=categories)
                else:
                    category = st.text_input("Category", value="Electronics")
                
                # Device dropdown
                if 'device' in label_encoders:
                    devices = label_encoders['device'].classes_.tolist()
                    device = st.selectbox("Device Used", options=devices)
                else:
                    device = st.selectbox("Device Used", options=["Mobile", "Desktop", "Tablet"])
                
                # Payment method dropdown
                if 'payment_method' in label_encoders:
                    payment_methods = label_encoders['payment_method'].classes_.tolist()
                    payment_method = st.selectbox("Payment Method", options=payment_methods)
                else:
                    payment_method = st.selectbox("Payment Method", options=["Credit Card", "Debit Card", "PayPal"])
            
            # Season dropdown (always include since it's in the error)
            col_season, col_extra = st.columns(2)
            
            with col_season:
                if 'season' in label_encoders:
                    seasons = label_encoders['season'].classes_.tolist()
                    season = st.selectbox("Season", options=seasons)
                else:
                    season = st.selectbox("Season", options=["Spring", "Summer", "Fall", "Winter"])
            
            with col_extra:
                # Add any other categorical features that might be in your model
                other_cats = [col for col in categorical_features if col not in ['category', 'device', 'payment_method', 'season']]
                if other_cats:
                    additional_inputs = {}
                    for cat_col in other_cats:
                        if cat_col in label_encoders:
                            options = label_encoders[cat_col].classes_.tolist()
                            additional_inputs[cat_col] = st.selectbox(f"{cat_col.replace('_', ' ').title()}", options=options)
            
            submitted = st.form_submit_button("Predict Return Probability", use_container_width=True)
    
    with col2:
        st.markdown("### Model Information")
        st.markdown(f"""
        <div class="metric-card">
            <strong>Model Type:</strong> Linear SVM with Calibration<br>
            <strong>Task:</strong> Binary Classification<br>
            <strong>Target:</strong> Product Return Prediction<br>
            <strong>Features Used:</strong><br>
            • Numeric ({len([f for f in numeric_features if f in globals()])}): Price, Discount, Final Price,<br>
            &nbsp;&nbsp;Review Count, Stock, Seller Rating,<br>
            &nbsp;&nbsp;Shipping Time<br>
            • Categorical ({len(categorical_features)}): {', '.join(categorical_features)}
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
                # Calculate final price (this matches training preprocessing)
                final_price = price * (1 - discount / 100)
                
                # Create a dictionary with all numeric features
                input_dict = {
                    'price': price,
                    'discount': discount,
                    'final_price': final_price,
                    'review_count': review_count,
                    'stock': stock,
                    'seller_rating': seller_rating,
                    'shipping_time_days': shipping_time
                }
                
                # Add categorical features
                input_dict['category'] = category
                input_dict['device'] = device
                input_dict['payment_method'] = payment_method
                input_dict['season'] = season
                
                # Add any additional categorical features
                if 'additional_inputs' in locals():
                    for cat_col, value in additional_inputs.items():
                        input_dict[cat_col] = value
                
                # Create DataFrame
                input_df = pd.DataFrame([input_dict])
                
                # Process all categorical features
                for col in categorical_features:
                    if col in input_df.columns and col in label_encoders:
                        # Convert to string
                        input_df[col] = input_df[col].astype(str)
                        # Handle unknown categories by mapping to the most common class
                        known_classes = set(label_encoders[col].classes_)
                        for idx, val in enumerate(input_df[col]):
                            if val not in known_classes:
                                # Use the first class as default
                                input_df.loc[idx, col] = label_encoders[col].classes_[0]
                        # Transform using label encoder
                        input_df[col] = label_encoders[col].transform(input_df[col])
                
                # Ensure all numeric features are present
                for col in numeric_features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training data order if feature_names is available
                if feature_names:
                    # Add any missing columns with default values
                    for col in feature_names:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    # Reorder
                    input_df = input_df[feature_names]
                else:
                    # Use the default order: numeric first, then categorical
                    input_df = input_df[numeric_features + categorical_features]
                
                # Make prediction
                prediction_proba = model.predict_proba(input_df)[0]
                prediction = model.predict(input_df)[0]
                
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
                        <h2> HIGH RETURN RISK</h2>
                        <p>This product has a {probability:.1f}% probability of being returned.</p>
                        <p><strong>Recommendation:</strong> Review product quality, improve descriptions, or optimize pricing.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif probability > 40:
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #fff3cd; border: 2px solid #ffc107;">
                        <h2>⚡ MODERATE RETURN RISK</h2>
                        <p>This product has a {probability:.1f}% probability of being returned.</p>
                        <p><strong>Recommendation:</strong> Monitor closely and consider slight improvements.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-low-risk">
                        <h2> LOW RETURN RISK</h2>
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
                        st.markdown("-  High price may increase return risk")
                    elif price < 20:
                        st.markdown("-  Low price generally reduces return risk")
                    else:
                        st.markdown("-  Price is in optimal range")
                    
                    if discount > 30:
                        st.markdown("-  High discount might indicate quality issues")
                    elif discount > 0:
                        st.markdown("-  Moderate discount is attractive")
                    else:
                        st.markdown("- ℹ No discount offered")
                    
                    if shipping_time > 7:
                        st.markdown("-  Long shipping time increases return likelihood")
                    elif shipping_time <= 3:
                        st.markdown("-  Fast shipping reduces return risk")
                    else:
                        st.markdown("-  Standard shipping time")
                
                with col_b:
                    st.markdown("**Quality Indicators:**")
                    
                    if seller_rating < 3.5:
                        st.markdown("-  Low seller rating increases return risk")
                    elif seller_rating >= 4.5:
                        st.markdown("-  High seller rating reduces returns")
                    else:
                        st.markdown("-  Acceptable seller rating")
                    
                    if review_count < 10:
                        st.markdown("-  Few reviews may indicate new product")
                    elif review_count > 100:
                        st.markdown("-  Many reviews suggest established product")
                    else:
                        st.markdown("-  Adequate number of reviews")
                    
                    if season in ['Winter', 'Fall']:
                        st.markdown("- ℹ Seasonal factor may influence returns")
                
                # Actionable insights
                st.markdown("###  Actionable Insights")
                
                if probability > 50:
                    insights = []
                    if discount > 20:
                        insights.append("- Consider reducing discount and improving product quality perception")
                    if seller_rating < 4.0:
                        insights.append("- Work on improving seller rating through better customer service")
                    if shipping_time > 5:
                        insights.append("- Optimize shipping processes or offer expedited shipping options")
                    if review_count < 20:
                        insights.append("- Encourage more customer reviews to build trust")
                    if price > 100:
                        insights.append("- Consider price optimization or bundle deals")
                    
                    if insights:
                        for insight in insights:
                            st.markdown(insight)
                    else:
                        st.markdown("- Review product details and consider quality improvements")
                else:
                    st.markdown("-  Current configuration is promising for low returns")
                    if discount < 20:
                        st.markdown("-  Consider small discounts to boost sales without increasing return risk")
                    st.markdown("-  Maintain high seller rating and fast shipping")
                    st.markdown("-  Continue gathering positive customer reviews")
                
                st.markdown("---")
                st.caption(" Disclaimer: This prediction is based on machine learning model analysis and should be used as a guideline, not an absolute guarantee.")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.error(f"Debug info - Available features: {list(input_df.columns)}")
                st.info("Please make sure all required features are provided.")

else:
    st.warning("""
    ###  Model Files Not Found
    
    To use this app, you need to first train and save the model by running `project.py`.
    
    Required files:
    - `svm_classification_model.pkl`
    - `label_encoders.pkl`
    
    Also recommended:
    - `feature_info.pkl` (saves feature names and types)
    
    Run your training script to generate these files, then restart the app.
    """)

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;">Built with Streamlit | Amazon Return Prediction Model</div>', unsafe_allow_html=True)
