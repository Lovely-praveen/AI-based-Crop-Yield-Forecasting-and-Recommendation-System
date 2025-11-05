import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

print("🌾 Starting Dataset Generation...")
print("=" * 60)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
print("✅ Directories created: models/, data/")

# Define parameters
n_samples = 5000
states = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka', 
          'Tamil Nadu', 'West Bengal', 'Madhya Pradesh', 'Gujarat', 'Rajasthan']
crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Soybean', 'Groundnut']
years = list(range(2010, 2024))

print(f"\n📊 Generating {n_samples} samples...")
print(f"   States: {len(states)}")
print(f"   Crops: {len(crops)}")
print(f"   Years: {years[0]}-{years[-1]}")

# Generate data
data = {
    'State': np.random.choice(states, n_samples),
    'Year': np.random.choice(years, n_samples),
    'Crop': np.random.choice(crops, n_samples),
    'Temperature': np.random.uniform(15, 40, n_samples),  # °C
    'Rainfall': np.random.uniform(50, 2500, n_samples),   # mm
    'Humidity': np.random.uniform(30, 90, n_samples),     # %
    'Soil_pH': np.random.uniform(4.5, 8.5, n_samples),
    'Nitrogen': np.random.uniform(20, 150, n_samples),    # kg/ha
    'Phosphorus': np.random.uniform(10, 80, n_samples),   # kg/ha
    'Potassium': np.random.uniform(10, 100, n_samples),   # kg/ha
    'Organic_Carbon': np.random.uniform(0.2, 2.5, n_samples),  # %
    'Area': np.random.uniform(1, 500, n_samples)          # hectares
}

df = pd.DataFrame(data)

# Create realistic yield calculations with crop-specific factors
def calculate_yield(row):
    """Calculate realistic crop yield based on conditions"""
    
    # Base yields by crop (tons/hectare)
    base_yields = {
        'Rice': 4.0, 'Wheat': 3.5, 'Cotton': 2.0, 'Sugarcane': 70.0,
        'Maize': 5.5, 'Soybean': 2.5, 'Groundnut': 1.8
    }
    
    base = base_yields.get(row['Crop'], 3.0)
    
    # Temperature factor (optimal ranges)
    temp_optimal = {'Rice': 28, 'Wheat': 22, 'Cotton': 30, 'Sugarcane': 32,
                    'Maize': 25, 'Soybean': 26, 'Groundnut': 28}
    temp_opt = temp_optimal.get(row['Crop'], 25)
    temp_factor = 1 - abs(row['Temperature'] - temp_opt) * 0.02
    temp_factor = max(0.5, min(1.2, temp_factor))
    
    # Rainfall factor
    rainfall_optimal = {'Rice': 1500, 'Wheat': 600, 'Cotton': 800, 'Sugarcane': 1800,
                        'Maize': 800, 'Soybean': 700, 'Groundnut': 600}
    rain_opt = rainfall_optimal.get(row['Crop'], 800)
    rain_factor = 1 - abs(row['Rainfall'] - rain_opt) / rain_opt * 0.5
    rain_factor = max(0.4, min(1.3, rain_factor))
    
    # Soil pH factor (optimal 6.0-7.5)
    ph_factor = 1.0
    if row['Soil_pH'] < 5.5:
        ph_factor = 0.7
    elif row['Soil_pH'] > 8.0:
        ph_factor = 0.75
    else:
        ph_factor = 1.1
    
    # NPK factor (nutrients)
    npk_factor = (
        (row['Nitrogen'] / 100) * 0.4 +
        (row['Phosphorus'] / 50) * 0.3 +
        (row['Potassium'] / 60) * 0.3
    )
    npk_factor = min(1.5, max(0.5, npk_factor))
    
    # Organic carbon factor
    oc_factor = 1 + (row['Organic_Carbon'] - 1) * 0.1
    oc_factor = max(0.8, min(1.2, oc_factor))
    
    # Calculate final yield with some random variation
    yield_value = base * temp_factor * rain_factor * ph_factor * npk_factor * oc_factor
    yield_value *= np.random.uniform(0.85, 1.15)  # Add natural variation
    
    return max(0.1, yield_value)  # Ensure positive values

print("\n🔄 Calculating yields...")
# Apply yield calculation
df['Production'] = df.apply(lambda row: calculate_yield(row) * row['Area'], axis=1)
df['Yield'] = df['Production'] / df['Area']

# Round numerical columns
df['Temperature'] = df['Temperature'].round(1)
df['Rainfall'] = df['Rainfall'].round(1)
df['Humidity'] = df['Humidity'].round(1)
df['Soil_pH'] = df['Soil_pH'].round(2)
df['Nitrogen'] = df['Nitrogen'].round(1)
df['Phosphorus'] = df['Phosphorus'].round(1)
df['Potassium'] = df['Potassium'].round(1)
df['Organic_Carbon'] = df['Organic_Carbon'].round(2)
df['Area'] = df['Area'].round(1)
df['Production'] = df['Production'].round(2)
df['Yield'] = df['Yield'].round(2)

# Save to CSV
df.to_csv('crop_yield_data.csv', index=False)

print("\n" + "=" * 60)
print("✅ DATASET CREATED SUCCESSFULLY!")
print("=" * 60)
print(f"\n📊 Dataset Statistics:")
print(f"   Total samples: {len(df):,}")
print(f"   Features: {len(df.columns)}")
print(f"   File: crop_yield_data.csv")
print(f"   Size: {os.path.getsize('crop_yield_data.csv') / 1024:.2f} KB")

print("\n🔍 Dataset Preview:")
print(df.head(10).to_string())

print("\n📈 Yield Statistics:")
print(df['Yield'].describe())

print("\n📋 Columns:")
for col in df.columns:
    print(f"   • {col}")

print("\n🎯 Next Step: Run 'python model_training.py'") 
