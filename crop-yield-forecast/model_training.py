 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def check_file_exists(filepath, error_msg):
    """Check if file exists and show helpful error"""
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: {error_msg}")
        print(f"   File not found: {filepath}")
        if 'csv' in filepath:
            print("\n💡 Solution: Run 'python generate_dataset.py' first")
        exit(1)

print("🚀 Starting Model Training Pipeline...")
print("=" * 60)

# Check if dataset exists
check_file_exists('crop_yield_data.csv', 'Dataset not found!')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)
print("✅ Models directory ready")

# Load dataset
print("\n📊 Loading dataset...")
try:
    df = pd.read_csv('crop_yield_data.csv')
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Check for required columns
required_cols = ['State', 'Crop', 'Year', 'Temperature', 'Rainfall', 'Humidity', 
                 'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon', 'Yield']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    exit(1)

# Data preprocessing
print("\n🔧 Preprocessing data...")

# Handle missing values
initial_rows = len(df)
df = df.dropna()
if len(df) < initial_rows:
    print(f"   ⚠️ Removed {initial_rows - len(df)} rows with missing values")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['State', 'Crop']

for col in categorical_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   ✓ Encoded {col}: {len(le.classes_)} unique values")

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("   ✓ Label encoders saved")

# Select features and target
feature_columns = ['State_encoded', 'Crop_encoded', 'Year', 'Temperature', 
                   'Rainfall', 'Humidity', 'Soil_pH', 'Nitrogen', 
                   'Phosphorus', 'Potassium', 'Organic_Carbon']

X = df[feature_columns]
y = df['Yield']

print(f"\n📈 Dataset Info:")
print(f"   Features: {len(feature_columns)}")
print(f"   Target: Yield (tons/hectare)")
print(f"   Samples: {len(X)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✂️ Data split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')
print("   ✓ Feature scaling applied and saved")

# Dictionary to store results
results = {}

print("\n" + "=" * 60)
print("🤖 TRAINING MODELS")
print("=" * 60)

# 1. Linear Regression
print("\n1️⃣ Training Linear Regression...")
try:
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    lr_r2 = r2_score(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    results['Linear Regression'] = {
        'R² Score': lr_r2,
        'MAE': lr_mae,
        'RMSE': lr_rmse,
        'model': lr_model
    }

    joblib.dump(lr_model, 'models/linear_regression_model.pkl')
    print(f"   ✅ R² Score: {lr_r2:.4f}")
    print(f"   ✅ MAE: {lr_mae:.4f}")
    print(f"   ✅ RMSE: {lr_rmse:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Random Forest
print("\n2️⃣ Training Random Forest Regressor...")
try:
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    results['Random Forest'] = {
        'R² Score': rf_r2,
        'MAE': rf_mae,
        'RMSE': rf_rmse,
        'model': rf_model
    }

    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print(f"   ✅ R² Score: {rf_r2:.4f}")
    print(f"   ✅ MAE: {rf_mae:.4f}")
    print(f"   ✅ RMSE: {rf_rmse:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. XGBoost
print("\n3️⃣ Training XGBoost Regressor...")
try:
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    results['XGBoost'] = {
        'R² Score': xgb_r2,
        'MAE': xgb_mae,
        'RMSE': xgb_rmse,
        'model': xgb_model
    }

    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    print(f"   ✅ R² Score: {xgb_r2:.4f}")
    print(f"   ✅ MAE: {xgb_mae:.4f}")
    print(f"   ✅ RMSE: {xgb_rmse:.4f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

if not results:
    print("\n❌ No models were trained successfully!")
    exit(1)

# Find best model
best_model_name = max(results, key=lambda x: results[x]['R² Score'])
best_model = results[best_model_name]['model']

print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[m]['R² Score'] for m in results.keys()],
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'RMSE': [results[m]['RMSE'] for m in results.keys()]
})
comparison_df = comparison_df.sort_values('R² Score', ascending=False)
print("\n" + comparison_df.to_string(index=False))

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {results[best_model_name]['R² Score']:.4f}")

# Save best model separately
joblib.dump(best_model, 'models/best_model.pkl')
with open('models/best_model_name.txt', 'w') as f:
    f.write(best_model_name)

# Save feature names for later use
with open('models/feature_names.txt', 'w') as f:
    f.write(','.join(feature_columns))

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETED!")
print("=" * 60)

print("\n📁 Saved Files:")
saved_files = [
    'models/linear_regression_model.pkl',
    'models/random_forest_model.pkl',
    'models/xgboost_model.pkl',
    'models/best_model.pkl',
    'models/scaler.pkl',
    'models/label_encoders.pkl',
    'models/best_model_name.txt',
    'models/feature_names.txt'
]

for f in saved_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"   ✓ {f} ({size:.1f} KB)")
    else:
        print(f"   ⚠️ {f} (not found)")

print("\n🎯 Next Step: Run 'streamlit run app.py'")
print("\n" + "=" * 60)