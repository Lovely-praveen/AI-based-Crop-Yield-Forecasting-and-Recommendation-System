import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌾 AI Crop Yield Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c5f2d;
    }
    h2 {
        color: #3d8b3d;
    }
    h3 {
        color: #4a9d4a;
    }
    </style>
    """, unsafe_allow_html=True)

# Check if models exist
def check_models_exist():
    required_files = [
        'models/linear_regression_model.pkl',
        'models/random_forest_model.pkl',
        'models/xgboost_model.pkl',
        'models/scaler.pkl',
        'models/label_encoders.pkl'
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    return len(missing) == 0, missing

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('models/linear_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        
        # Load best model name if exists
        if os.path.exists('models/best_model_name.txt'):
            with open('models/best_model_name.txt', 'r') as f:
                best_model_name = f.read().strip()
        else:
            best_model_name = 'Random Forest'
        
        return {
            'Linear Regression': lr_model,
            'Random Forest': rf_model,
            'XGBoost': xgb_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'best_model_name': best_model_name
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# Load dataset
@st.cache_data
def load_data():
    try:
        if not os.path.exists('crop_yield_data.csv'):
            st.error("❌ Dataset file 'crop_yield_data.csv' not found!")
            st.info("💡 Please run: python generate_dataset.py")
            return None
        df = pd.read_csv('crop_yield_data.csv')
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

# Recommendation function
def get_recommendations(inputs):
    recommendations = []
    
    # Nitrogen recommendations
    if inputs['Nitrogen'] < 40:
        recommendations.append("🔴 **Low Nitrogen**: Apply urea fertilizer (46-0-0) at 100-150 kg/ha to boost nitrogen levels.")
    elif inputs['Nitrogen'] > 120:
        recommendations.append("🟡 **High Nitrogen**: Reduce nitrogen application to prevent lodging and environmental damage.")
    else:
        recommendations.append("✅ **Nitrogen Optimal**: Current nitrogen levels are good.")
    
    # Phosphorus recommendations
    if inputs['Phosphorus'] < 20:
        recommendations.append("🔴 **Low Phosphorus**: Apply DAP (18-46-0) at 50-75 kg/ha for root development.")
    elif inputs['Phosphorus'] > 60:
        recommendations.append("🟡 **High Phosphorus**: Reduce phosphorus to prevent nutrient imbalance.")
    else:
        recommendations.append("✅ **Phosphorus Optimal**: Phosphorus levels are adequate.")
    
    # Potassium recommendations
    if inputs['Potassium'] < 25:
        recommendations.append("🔴 **Low Potassium**: Apply MOP (0-0-60) at 40-60 kg/ha for better crop quality.")
    elif inputs['Potassium'] > 80:
        recommendations.append("🟡 **High Potassium**: Current levels are high, reduce application.")
    else:
        recommendations.append("✅ **Potassium Optimal**: Potassium is at good levels.")
    
    # Soil pH recommendations
    if inputs['Soil_pH'] < 5.5:
        recommendations.append("🔴 **Acidic Soil**: Apply agricultural lime (2-3 tons/ha) to raise pH to 6.0-7.0.")
    elif inputs['Soil_pH'] > 8.0:
        recommendations.append("🔴 **Alkaline Soil**: Add sulfur or gypsum to lower pH to 6.5-7.5.")
    else:
        recommendations.append("✅ **pH Optimal**: Soil pH is in the ideal range (6.0-7.5).")
    
    # Rainfall recommendations
    if inputs['Rainfall'] < 400:
        recommendations.append("🔴 **Low Rainfall**: Install drip irrigation or use drought-tolerant crop varieties.")
    elif inputs['Rainfall'] > 2000:
        recommendations.append("🟡 **High Rainfall**: Ensure proper drainage to prevent waterlogging.")
    else:
        recommendations.append("✅ **Rainfall Adequate**: Rainfall is sufficient for crop growth.")
    
    # Temperature recommendations
    if inputs['Temperature'] < 18:
        recommendations.append("🟡 **Low Temperature**: Consider cold-tolerant varieties or adjust planting season.")
    elif inputs['Temperature'] > 35:
        recommendations.append("🟡 **High Temperature**: Use heat-tolerant varieties and ensure adequate irrigation.")
    else:
        recommendations.append("✅ **Temperature Optimal**: Temperature range is suitable.")
    
    # Organic Carbon recommendations
    if inputs['Organic_Carbon'] < 0.5:
        recommendations.append("🔴 **Low Organic Matter**: Add compost or farmyard manure (10-15 tons/ha).")
    else:
        recommendations.append("✅ **Organic Carbon Good**: Soil organic matter is healthy.")
    
    return recommendations

# Prediction function
def make_prediction(model, inputs, scaler, label_encoders):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'State_encoded': [label_encoders['State'].transform([inputs['State']])[0]],
            'Crop_encoded': [label_encoders['Crop'].transform([inputs['Crop']])[0]],
            'Year': [inputs['Year']],
            'Temperature': [inputs['Temperature']],
            'Rainfall': [inputs['Rainfall']],
            'Humidity': [inputs['Humidity']],
            'Soil_pH': [inputs['Soil_pH']],
            'Nitrogen': [inputs['Nitrogen']],
            'Phosphorus': [inputs['Phosphorus']],
            'Potassium': [inputs['Potassium']],
            'Organic_Carbon': [inputs['Organic_Carbon']]
        })
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return prediction, input_scaled
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Main app
def main():
    # Check if models exist
    models_exist, missing_files = check_models_exist()
    
    if not models_exist:
        st.error("❌ Required model files are missing!")
        st.warning("Missing files:")
        for f in missing_files:
            st.write(f"   • {f}")
        st.info("💡 **Solution**: Run the following commands in order:")
        st.code("python generate_dataset.py\npython model_training.py", language="bash")
        st.stop()
    
    # Load resources
    models_dict = load_models()
    df = load_data()
    
    if models_dict is None or df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🌾 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🎯 Predict Yield", "📊 Visualizations", "🤖 Model Insights", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    # HOME PAGE
    if page == "🏠 Home":
        st.title("🌾 AI-Based Crop Yield Forecasting System")
        st.markdown("### Welcome to the Smart Agriculture Platform")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Accurate Predictions")
            st.write("Get precise yield forecasts using advanced ML models")
        
        with col2:
            st.markdown("#### 💡 Smart Recommendations")
            st.write("Receive actionable insights to improve crop yield")
        
        with col3:
            st.markdown("#### 📈 Data Visualization")
            st.write("Explore trends and patterns in agricultural data")
        
        st.markdown("---")
        
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Crops Covered", df['Crop'].nunique())
        with col3:
            st.metric("States", df['State'].nunique())
        with col4:
            st.metric("Years", f"{df['Year'].min()}-{df['Year'].max()}")
        
        st.markdown("### 🚀 Quick Start")
        st.info("👈 Use the sidebar to navigate to **Predict Yield** and start forecasting!")
        
        # Display sample data
        st.markdown("### 📋 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    # PREDICT YIELD PAGE
    elif page == "🎯 Predict Yield":
        st.title("🎯 Crop Yield Prediction")
        st.markdown("Enter crop and environmental parameters to predict yield")
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                state = st.selectbox("📍 Select State", sorted(df['State'].unique()))
                crop = st.selectbox("🌾 Select Crop", sorted(df['Crop'].unique()))
                year = st.number_input("📅 Year", min_value=2010, max_value=2030, value=2024)
                temperature = st.number_input("🌡️ Average Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.1)
            
            with col2:
                rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=3000.0, value=800.0, step=10.0)
                humidity = st.number_input("💧 Humidity (%)", min_value=20.0, max_value=100.0, value=65.0, step=1.0)
                soil_ph = st.number_input("⚗️ Soil pH", min_value=3.0, max_value=10.0, value=6.5, step=0.1)
                nitrogen = st.number_input("🧪 Nitrogen (kg/ha)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
            
            with col3:
                phosphorus = st.number_input("🧪 Phosphorus (kg/ha)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
                potassium = st.number_input("🧪 Potassium (kg/ha)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
                organic_carbon = st.number_input("🌱 Organic Carbon (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            
            submitted = st.form_submit_button("🔮 Predict Yield", use_container_width=True)
        
        if submitted:
            # Prepare inputs
            inputs = {
                'State': state,
                'Crop': crop,
                'Year': year,
                'Temperature': temperature,
                'Rainfall': rainfall,
                'Humidity': humidity,
                'Soil_pH': soil_ph,
                'Nitrogen': nitrogen,
                'Phosphorus': phosphorus,
                'Potassium': potassium,
                'Organic_Carbon': organic_carbon
            }
            
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Make predictions with all models
            results = {}
            for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
                model = models_dict[model_name]
                pred, input_scaled = make_prediction(
                    model, inputs, models_dict['scaler'], models_dict['label_encoders']
                )
                if pred is not None:
                    results[model_name] = pred
            
            if not results:
                st.error("Failed to make predictions. Please check your inputs.")
                st.stop()
            
            # Display predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 📈 Linear Regression")
                st.metric("Predicted Yield", f"{results['Linear Regression']:.2f} tons/ha")
            
            with col2:
                st.markdown("### 🌲 Random Forest")
                st.metric("Predicted Yield", f"{results['Random Forest']:.2f} tons/ha")
            
            with col3:
                st.markdown("### ⚡ XGBoost")
                st.metric("Predicted Yield", f"{results['XGBoost']:.2f} tons/ha")
            
            # Best model prediction
            best_model_name = models_dict['best_model_name']
            best_prediction = results[best_model_name]
            
            st.success(f"🏆 **Best Model ({best_model_name})**: {best_prediction:.2f} tons/ha")
            
            # Confidence indicator
            avg_pred = np.mean(list(results.values()))
            std_pred = np.std(list(results.values()))
            confidence = max(0, min(100, 100 - (std_pred / avg_pred * 100) if avg_pred > 0 else 0))
            
            st.info(f"📊 **Prediction Confidence**: {confidence:.1f}%")
            
            # Recommendations
            st.markdown("---")
            st.markdown("## 💡 Recommendations for Yield Improvement")
            
            recommendations = get_recommendations(inputs)
            
            for rec in recommendations:
                if "🔴" in rec:
                    st.error(rec)
                elif "🟡" in rec:
                    st.warning(rec)
                else:
                    st.success(rec)
    
    # VISUALIZATIONS PAGE
    elif page == "📊 Visualizations":
        st.title("📊 Data Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🌾 Crop Analysis", "🗺️ Regional Analysis", "🔗 Correlations"])
        
        with tab1:
            st.markdown("### 📈 Yield Trends Over Years")
            
            # Yield by year
            yearly_yield = df.groupby('Year')['Yield'].mean().reset_index()
            
            fig = px.line(yearly_yield, x='Year', y='Yield', 
                         title='Average Crop Yield Over Years',
                         markers=True)
            fig.update_layout(xaxis_title="Year", yaxis_title="Yield (tons/ha)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Rainfall vs Yield
            st.markdown("### 🌧️ Rainfall vs Yield")
            sample_data = df.sample(min(1000, len(df)))
            
            fig = px.scatter(sample_data, x='Rainfall', y='Yield', 
                           color='Crop', title='Rainfall Impact on Yield',
                           opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🌾 Crop-wise Analysis")
            
            crop_yield = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).reset_index()
            
            fig = px.bar(crop_yield, x='Crop', y='Yield',
                        title='Average Yield by Crop Type',
                        color='Yield',
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
            
            # Crop distribution
            crop_counts = df['Crop'].value_counts().reset_index()
            crop_counts.columns = ['Crop', 'Count']
            
            fig = px.pie(crop_counts, values='Count', names='Crop',
                        title='Crop Distribution in Dataset')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🗺️ State-wise Yield Analysis")
            
            state_yield = df.groupby('State')['Yield'].mean().sort_values(ascending=False).reset_index()
            
            fig = px.bar(state_yield, x='State', y='Yield',
                        title='Average Yield by State',
                        color='Yield',
                        color_continuous_scale='Blues')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 🔗 Feature Correlations")
            
            # Correlation heatmap
            numeric_cols = ['Temperature', 'Rainfall', 'Humidity', 'Soil_pH', 
                          'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon', 'Yield']
            corr_data = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax)
            plt.title('Feature Correlation Matrix')
            st.pyplot(fig, use_container_width=True)
            plt.close()
    
    # MODEL INSIGHTS PAGE
    elif page == "🤖 Model Insights":
        st.title("🤖 Model Performance Insights")
        
        # Load dataset for evaluation
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        df_copy = df.copy()
        le_state = models_dict['label_encoders']['State']
        le_crop = models_dict['label_encoders']['Crop']
        
        df_copy['State_encoded'] = le_state.transform(df_copy['State'])
        df_copy['Crop_encoded'] = le_crop.transform(df_copy['Crop'])
        
        feature_columns = ['State_encoded', 'Crop_encoded', 'Year', 'Temperature', 
                          'Rainfall', 'Humidity', 'Soil_pH', 'Nitrogen', 
                          'Phosphorus', 'Potassium', 'Organic_Carbon']
        
        X = df_copy[feature_columns]
        y = df_copy['Yield']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = models_dict['scaler'].transform(X_test)
        
        # Evaluate models
        st.markdown("### 📊 Model Comparison")
        
        comparison_data = []
        for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
            model = models_dict[model_name]
            predictions = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            
            comparison_data.append({
                'Model': model_name,
                'R² Score': f"{r2:.4f}",
                'MAE': f"{mae:.4f}",
                'RMSE': f"{rmse:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Feature importance for Random Forest
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance (Random Forest)")
        
        rf_model = models_dict['Random Forest']
        feature_names_display = ['State', 'Crop', 'Year', 'Temperature', 
                                 'Rainfall', 'Humidity', 'Soil pH', 'Nitrogen', 
                                 'Phosphorus', 'Potassium', 'Organic Carbon']
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names_display,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(feature_importance, x='Importance', y='Feature',
                    orientation='h',
                    title='Feature Importance Rankings',
                    color='Importance',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Model predictions vs actual
        st.markdown("---")
        st.markdown("### 📈 Predictions vs Actual Values")
        
        selected_model = st.selectbox("Select Model", ['Linear Regression', 'Random Forest', 'XGBoost'])
        model = models_dict[selected_model]
        predictions = model.predict(X_test_scaled)
        
        # Sample for visualization
        sample_size = min(500, len(y_test))
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test_array[indices], y=predictions[indices],
                                mode='markers',
                                name='Predictions',
                                marker=dict(size=5, opacity=0.6)))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'{selected_model}: Predicted vs Actual Yield',
                         xaxis_title='Actual Yield (tons/ha)',
                         yaxis_title='Predicted Yield (tons/ha)')
        st.plotly_chart(fig, use_container_width=True)
    
    # ABOUT PAGE
    elif page == "ℹ️ About":
        st.title("ℹ️ About This Project")
        
        st.markdown("""
        ### 🌾 AI-Based Crop Yield Forecasting System
        
        This comprehensive platform leverages machine learning to predict agricultural crop yields 
        and provide actionable recommendations for farmers and agricultural planners.
        
        #### 🎯 Key Features:
        - **Multi-Model Prediction**: Compare Linear Regression, Random Forest, and XGBoost
        - **Smart Recommendations**: Get personalized advice based on soil and weather conditions
        - **Interactive Visualizations**: Explore trends and patterns in agricultural data
        - **Model Performance**: Understand model accuracy and feature importance
        
        #### 📊 Dataset Information:
        - **Crops Covered**: Rice, Wheat, Cotton, Sugarcane, Maize, Soybean, Groundnut
        - **States**: 10 major agricultural states in India
        - **Time Period**: 2010-2024
        - **Features**: 11 environmental and soil parameters
        
        #### 🤖 Machine Learning Models:
        1. **Linear Regression**: Simple baseline model
        2. **Random Forest**: Ensemble method with high accuracy
        3. **XGBoost**: Gradient boosting for optimal performance
        
        #### 🛠️ Technologies Used:
        - **Frontend**: Streamlit
        - **ML Libraries**: Scikit-learn, XGBoost
        - **Visualization**: Plotly, Matplotlib, Seaborn
        
        #### 📧 Contact:
        For questions or feedback, please reach out to your agricultural extension office.
        
        ---
        
        **Version**: 1.0  
        **Last Updated**: 2024  
        **License**: MIT
        """)

if __name__ == "__main__":
    main() 
