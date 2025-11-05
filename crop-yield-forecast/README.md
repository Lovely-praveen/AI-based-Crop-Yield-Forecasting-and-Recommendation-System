# 🌾 AI-Based Crop Yield Forecasting System

A complete machine learning system for predicting crop yields and providing smart agricultural recommendations.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Model Training](#model-training)
- [Running the App](#running-the-app)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)

## ✨ Features

- 🎯 **Multi-Model Predictions**: Compare Linear Regression, Random Forest, and XGBoost
- 💡 **Smart Recommendations**: Get actionable advice for yield improvement
- 📊 **Interactive Visualizations**: Explore trends with Plotly charts
- 🔍 **Explainable AI**: Understand predictions with SHAP values
- 🌈 **Beautiful UI**: Sky-blue themed Streamlit interface
- 📈 **Model Comparison**: View performance metrics for all models
- 🗺️ **Regional Analysis**: Compare yields across states and crops

## 🚀 Installation

### Step 1: Clone or Download Project

```bash
# Create project directory
mkdir crop-yield-forecast
cd crop-yield-forecast
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap
- joblib
- plotly

## ⚡ Quick Start

### 1. Generate Dataset

```bash
python generate_dataset.py
```

This creates `crop_yield_data.csv` with 5,000 samples.

### 2. Train Models

```bash
python model_training.py
```

This will:
- Train 3 ML models (Linear Regression, Random Forest, XGBoost)
- Save models in `models/` folder
- Display performance metrics
- Save the best model

**Expected Output:**
```
✅ MODEL TRAINING COMPLETED!
📊 MODEL COMPARISON
Model                R² Score    MAE      RMSE
Random Forest        0.9234      0.8765   1.2345
XGBoost             0.9156      0.9012   1.3012
Linear Regression    0.7823      1.4532   2.1234

🏆 Best Model: Random Forest
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
crop-yield-forecast/
│
├── app.py                          # Streamlit frontend
├── model_training.py               # Model training script
├── generate_dataset.py             # Dataset generation
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
│
├── models/                         # Trained models
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── best_model_name.txt
│
└── crop_yield_data.csv            # Dataset
```

## 📊 Dataset Information

### Columns

| Column | Description | Range |
|--------|-------------|-------|
| State | Indian state | 10 states |
| Year | Year of cultivation | 2010-2024 |
| Crop | Crop type | 7 crops |
| Temperature | Average temperature | 15-40°C |
| Rainfall | Annual rainfall | 50-2500mm |
| Humidity | Average humidity | 30-90% |
| Soil_pH | Soil pH level | 4.5-8.5 |
| Nitrogen | Nitrogen content | 20-150 kg/ha |
| Phosphorus | Phosphorus content | 10-80 kg/ha |
| Potassium | Potassium content | 10-100 kg/ha |
| Organic_Carbon | Organic carbon | 0.2-2.5% |
| Area | Cultivated area | 1-500 ha |
| Production | Total production | Calculated |
| Yield | Yield per hectare | Target variable |

### Crops Covered
- Rice
- Wheat
- Cotton
- Sugarcane
- Maize
- Soybean
- Groundnut

### States Covered
- Punjab
- Haryana
- Uttar Pradesh
- Maharashtra
- Karnataka
- Tamil Nadu
- West Bengal
- Madhya Pradesh
- Gujarat
- Rajasthan

## 🤖 Model Training Details

### Data Preprocessing
1. Load dataset and handle missing values
2. Encode categorical variables (State, Crop)
3. Split data (80% train, 20% test)
4. Scale features using StandardScaler

### Models Trained

#### 1. Linear Regression
- Simple baseline model
- Fast training and prediction
- Good for linear relationships

#### 2. Random Forest Regressor
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

#### 3. XGBoost Regressor
```python
XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Evaluation Metrics
- **R² Score**: Model accuracy (0-1, higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

## 🎨 Running the App

### Main Pages

#### 1. Home 🏠
- Dataset overview
- Quick statistics
- Sample data preview

#### 2. Predict Yield 🎯
- Input form for crop parameters
- Real-time predictions from all models
- Smart recommendations
- SHAP explainability

#### 3. Visualizations 📊
- Yield trends over years
- Crop-wise analysis
- Regional comparisons
- Correlation heatmaps

#### 4. Model Insights 🤖
- Model performance comparison
- Feature importance rankings
- Prediction vs Actual plots

#### 5. About ℹ️
- Project information
- Technologies used
- Contact details

## 🔧 Customization

### Adding New Crops
1. Update `crops` list in `generate_dataset.py`
2. Add crop-specific parameters
3. Regenerate dataset
4. Retrain models

### Changing Model Parameters
Edit `model_training.py`:

```python
rf_model = RandomForestRegressor(
    n_estimators=200,  # Increase trees
    max_depth=20,      # Increase depth
    random_state=42
)
```

### UI Customization
Edit CSS in `app.py`:

```python
st.markdown("""
    <style>
    .main {
        background-color: #your-color;
    }
    </style>
""", unsafe_allow_html=True)
```

## 🐛 Troubleshooting

### Issue: "Models not found"
**Solution**: Run `python model_training.py` first

### Issue: "Dataset not found"
**Solution**: Run `python generate_dataset.py` first

### Issue: SHAP visualization error
**Solution**: This is normal for some model types, the app handles it gracefully

### Issue: Port already in use
**Solution**: Use a different port:
```bash
streamlit run app.py --server.port 8502
```

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Hugging Face Spaces
1. Create new Space
2. Upload files
3. Add `requirements.txt`
4. Deploy

## 📝 Data Sources

This project uses synthetic data generated to simulate:
- [Crop Production in India - Kaggle](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india)
- [Rainfall in India - Kaggle](https://www.kaggle.com/datasets/rajanand/rainfall-in-india)
- [Soil Data India - Kaggle](https://www.kaggle.com/datasets)
- [FAO Crop Yield Dataset](http://www.fao.org/faostat/)

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: SHAP
- **Model Storage**: Joblib

## 📄 License

MIT License - feel free to use and modify!

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

## 📧 Contact

For questions or feedback:
- Create an issue on GitHub
- Email: your-email@example.com

---

**Made with ❤️ for Smart Agriculture** 
