import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv(r"C:\Users\ahmed\Desktop\data computation\amazon_ecommerce_1M.csv")
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Store original for later processing
df_original = df.copy()

# =========================
# REMOVE LEAKY FEATURES (DATA LEAKAGE)
# =========================
print("\n" + "="*60)
print("REMOVING LEAKY FEATURES")
print("="*60)

# Columns that cause data leakage - these should NEVER be used for training
leaky_columns = [
    # User identifiers
    'user_id', 'User ID', 'userid', 'customer_id', 'CustomerID', 'customer',
    'user', 'buyer_id', 'shopper_id',
    
    # Product identifiers (unique per product, not predictive)
    'product_id', 'Product ID', 'productid', 'asin', 'sku', 'item_id',
    'product_code', 'upc', 'ean', 'isbn',
    
    # Seller identifiers
    'seller_id', 'Seller ID', 'sellerid', 'vendor_id', 'vendor',
    'merchant_id', 'supplier_id',
    
    # Transaction identifiers
    'transaction_id', 'order_id', 'receipt_id', 'invoice_id',
    'purchase_id', 'cart_id', 'session_id',
    
    # Date/time fields (use engineered features instead)
    'purchase_date', 'order_date', 'date', 'timestamp', 'created_at',
    'updated_at', 'datetime', 'order_timestamp',
    
    # Location data (user-specific, not product-specific)
    'location', 'city', 'state', 'zip', 'postal_code', 'country',
    'address', 'shipping_address', 'billing_address', 'geo_location',
    
    # Session and tracking data
    'ip_address', 'user_agent', 'browser', 'browser_version',
    'os', 'device_id', 'cookie_id', 'session_duration',
    
    # User behavior that happens AFTER purchase (target leakage)
    'viewed_before', 'clicked', 'searched', 'browsing_history',
    'cart_add_time', 'checkout_time'
]

# Count how many leaky columns exist
leaky_cols_present = [col for col in leaky_columns if col in df.columns]
print(f"Found {len(leaky_cols_present)} leaky columns in dataset")

# Remove leaky columns from the dataframe BEFORE splitting features
df = df.drop(columns=[col for col in leaky_columns if col in df.columns], errors='ignore')
print(f"Removed {len([col for col in leaky_columns if col in df_original.columns])} leaky columns")
print(f"Remaining columns: {df.shape[1]}")

# =========================
# EDA SECTION (using sample for speed)
# =========================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Take sample for EDA
eda_sample = df.sample(min(50000, len(df)), random_state=42)

# Check target distribution
print("\nTarget Distribution:")
print(df['is_returned'].value_counts(normalize=True))

numeric_cols_eda = ['price', 'discount', 'final_price', 'review_count', 'stock', 'seller_rating', 'shipping_time_days']

# Figure 1: Overview
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

sns.countplot(x='is_returned', data=eda_sample, ax=axes[0])
axes[0].set_title('Return Distribution')

(df.isnull().mean() * 100).sort_values().plot(kind='barh', ax=axes[1])
axes[1].set_title('Missing Values (%)')

available_numeric = [col for col in numeric_cols_eda if col in eda_sample.columns]
if 'is_returned' in eda_sample.columns:
    corr_cols = available_numeric + ['is_returned']
    if all(col in eda_sample.columns for col in corr_cols):
        sns.heatmap(eda_sample[corr_cols].corr(), annot=True, cmap='coolwarm', ax=axes[2])
axes[2].set_title('Correlation Matrix')

if 'category' in eda_sample.columns:
    df.groupby('category')['is_returned'].mean().sort_values().plot(kind='barh', ax=axes[3])
    axes[3].set_title('Return Rate by Category')
    
    df['category'].value_counts().head(10).plot(kind='barh', ax=axes[4])
    axes[4].set_title('Top 10 Categories')

if 'shipping_time_days' in eda_sample.columns:
    sns.boxplot(x='is_returned', y='shipping_time_days', data=eda_sample, ax=axes[5])
    axes[5].set_title('Shipping Time vs Return')

plt.tight_layout()
plt.show()

# Figure 2: Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(available_numeric[:6]):
    if col in eda_sample.columns:
        sns.kdeplot(data=eda_sample, x=col, hue='is_returned', fill=True, ax=axes[i])
        axes[i].set_title(f'{col} vs Return')

plt.tight_layout()
plt.show()

# Figure 3: Product category analysis (removed device and payment as they're user-specific)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

if 'subcategory' in eda_sample.columns:
    sns.barplot(x='subcategory', y='is_returned', data=eda_sample, ax=axes[0])
    axes[0].set_title('Return Rate by Subcategory')
    axes[0].tick_params(axis='x', rotation=45)

if 'brand' in eda_sample.columns:
    top_brands = eda_sample.groupby('brand')['is_returned'].mean().sort_values().head(10)
    top_brands.plot(kind='barh', ax=axes[1])
    axes[1].set_title('Top 10 Brands by Return Rate')

plt.tight_layout()
plt.show()

# =========================
# DATA PREPROCESSING (OPTIMIZED)
# =========================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicates")

# Handle outliers conservatively
print("\nHandling outliers...")
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# Exclude target from outlier handling
numeric_cols = [col for col in numeric_cols if col != 'is_returned']
initial_rows = len(df)

for col in numeric_cols:
    Q1 = df[col].quantile(0.01)
    Q3 = df[col].quantile(0.99)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Removed {initial_rows - len(df)} outliers")
print(f"Final dataset shape: {df.shape}")

# Separate features and target
if 'is_returned' not in df.columns:
    raise ValueError("Target column 'is_returned' not found")
    
X = df.drop('is_returned', axis=1)
y = df['is_returned']

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")
if categorical_features:
    print(f"Categorical features: {categorical_features[:5]}")

# Encode categorical variables
print("\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  Encoded {col}: {len(le.classes_)} categories")

# Handle missing values
print("\nHandling missing values...")
print(f"Missing values before imputation: {X.isnull().sum().sum()}")

numeric_imputer = SimpleImputer(strategy='median')
X_numeric = pd.DataFrame(
    numeric_imputer.fit_transform(X[numeric_features]),
    columns=numeric_features,
    index=X.index
)

if categorical_features:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X_categorical = pd.DataFrame(
        categorical_imputer.fit_transform(X[categorical_features]),
        columns=categorical_features,
        index=X.index
    )
    X = pd.concat([X_numeric, X_categorical], axis=1)
else:
    X = X_numeric

print(f"Missing values after imputation: {X.isnull().sum().sum()}")

# =========================
# TRAIN TEST SPLIT
# =========================
print("\n" + "="*60)
print("TRAIN TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")

# =========================
# OPTIMIZED SVM MODEL BUILDING
# =========================
print("\n" + "="*60)
print("OPTIMIZED SVM CLASSIFICATION MODEL")
print("="*60)

# Create pipeline with LinearSVC (much faster than SVC)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(random_state=42, class_weight='balanced', dual='auto', max_iter=1000))
])

# Use smaller sample for tuning (reduced from 50k to 20k)
use_sample_for_tuning = len(X_train) > 20000
if use_sample_for_tuning:
    print(f"\nUsing sample for hyperparameter tuning (20,000 samples for speed)")
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, train_size=20000, 
        random_state=42, stratify=y_train
    )
else:
    X_train_sample, y_train_sample = X_train, y_train

# Minimal hyperparameter grid for speed
param_grid = {
    'svm__C': [0.1, 1, 10]
}

print("\nPerforming grid search with reduced combinations...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='f1_weighted',
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train_sample, y_train_sample)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Add probability calibration to the best model
print("\nAdding probability calibration...")
best_svm = grid_search.best_estimator_
calibrated_model = CalibratedClassifierCV(best_svm, cv=3)

# Train final model on full training set
print("\nTraining final model on full dataset...")
calibrated_model.fit(X_train, y_train)

# Cross-validation on full training (reduced folds for speed)
print("\nPerforming 3-fold cross-validation...")
cv_scores = cross_val_score(calibrated_model, X_train, y_train, cv=3, scoring='f1_weighted')
print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =========================
# MODEL EVALUATION
# =========================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

y_pred = calibrated_model.predict(X_test)
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Returned', 'Returned']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Negatives: {cm[0,0]:,} | False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,} | True Positives: {cm[1,1]:,}")

# =========================
# VISUALIZATIONS
# =========================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xticklabels(['Not Returned', 'Returned'])
axes[0, 0].set_yticklabels(['Not Returned', 'Returned'])

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc_roc:.3f})', linewidth=2)
axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Random Classifier', linewidth=1)
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Prediction Probability Distribution
axes[1, 0].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Not Returned', color='green', edgecolor='black')
axes[1, 0].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Returned', color='red', edgecolor='black')
axes[1, 0].set_xlabel('Predicted Probability of Return')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Probability Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
values = [accuracy, precision, recall, f1, auc_roc]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
bars = axes[1, 1].bar(metrics, values, color=colors, edgecolor='black')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Performance Metrics')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# =========================
# FEATURE ANALYSIS (for linear kernel)
# =========================
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# LinearSVC always has coefficients
coefficients = np.abs(calibrated_model.estimator.named_steps['svm'].coef_[0])
feature_names = X.columns

# Get top 15 features
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': coefficients
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='teal')
plt.xlabel('Absolute Coefficient')
plt.title('Top 15 Most Important Features (Linear SVM)')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nTop 10 important features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# =========================
# BUSINESS INSIGHTS
# =========================
print("\n" + "="*60)
print("BUSINESS INSIGHTS")
print("="*60)

tn, fp, fn, tp = cm.ravel()
total_predictions = len(y_test)

print(f"\nTotal test samples: {total_predictions:,}")
print(f"Correct predictions: {tn + tp:,} ({(tn+tp)/total_predictions*100:.1f}%)")
print(f"Incorrect predictions: {fp + fn:,} ({(fp+fn)/total_predictions*100:.1f}%)")

# Cost analysis
cost_false_positive = 10
cost_false_negative = 50

estimated_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
max_possible_cost = (fp + fn) * max(cost_false_positive, cost_false_negative)
cost_savings_percentage = (1 - estimated_cost / max_possible_cost) * 100 if max_possible_cost > 0 else 0

print(f"\nEstimated cost with current model: ${estimated_cost:,}")
print(f"Cost savings: {cost_savings_percentage:.1f}% vs worst case")

# Return rate insights
actual_return_rate = y_test.mean()
predicted_return_rate = y_pred.mean()
print(f"\nActual return rate: {actual_return_rate:.2%}")
print(f"Predicted return rate: {predicted_return_rate:.2%}")

# Model performance by class
print(f"\nPerformance on 'Not Returned' class: {cm[0,0]/(cm[0,0]+cm[0,1]):.2%} accuracy")
print(f"Performance on 'Returned' class: {cm[1,1]/(cm[1,0]+cm[1,1]):.2%} accuracy")

# =========================
# SAVE MODEL
# =========================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

import joblib
joblib.dump(calibrated_model, 'svm_classification_model.pkl')
print("Model saved as 'svm_classification_model.pkl'")

# Save the preprocessors
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(numeric_imputer, 'numeric_imputer.pkl')
if categorical_features:
    joblib.dump(categorical_imputer, 'categorical_imputer.pkl')
print("Preprocessing objects saved")

print("\n" + "="*60)
print("MODEL BUILDING COMPLETE")
print("="*60)
print("\nSummary:")
print(f"  - Problem type: Binary Classification")
print(f"  - Target variable: is_returned")
print(f"  - Best parameters: {grid_search.best_params_}")
print(f"  - Test F1-Score: {f1:.4f}")
print(f"  - Test AUC-ROC: {auc_roc:.4f}")
print(f"  - Final features used: {len(X.columns)} product-only features")

# =========================
# SAVE ADDITIONAL FEATURE INFO
# =========================
print("\n" + "="*60)
print("SAVING FEATURE INFORMATION")
print("="*60)

# Save feature names and types for the Streamlit app
feature_info = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'all_features': X.columns.tolist(),
    'feature_count': len(X.columns)
}
joblib.dump(feature_info, 'feature_info.pkl')
print(f"Feature info saved with {len(X.columns)} total features")
print(f"  - Numeric features: {len(numeric_features)}")
print(f"  - Categorical features: {len(categorical_features)}")
print(f"\nFinal feature list (product-only):")
for i, feat in enumerate(X.columns[:10]):
    print(f"  {i+1}. {feat}")
if len(X.columns) > 10:
    print(f"  ... and {len(X.columns)-10} more features")
