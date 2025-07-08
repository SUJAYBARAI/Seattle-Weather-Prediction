import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

# Set page config
st.set_page_config(page_title="Weather Prediction", page_icon="⛅", layout="wide")

# Load and preprocess data
@st.cache_data
def load_weather_data():
    weather_data = pd.read_csv("seattle-weather.csv")
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    weather_data['year'] = weather_data['date'].dt.year
    weather_data['month'] = weather_data['date'].dt.month
    return weather_data

weather_data = load_weather_data()

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    """
    This application predicts weather conditions based on:
    - Maximum temperature (°C)
    - Minimum temperature (°C)
    - Precipitation (inches)
    
    Using a Random Forest classifier trained on Seattle weather data.
    """
)

# Main content
st.title("⛅ Seattle Weather Prediction Analytics")
st.write("Explore statistical insights and predict weather conditions based on historical data.")

# Show raw data if checkbox is checked
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(weather_data)

# Train model (cached)
@st.cache_resource
def build_weather_model():
    features = ['temp_max', 'temp_min', 'precipitation']
    X = weather_data[features]
    y = weather_data['weather']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    classifier.fit(X_train_scaled, y_train)
    
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return classifier, scaler, label_encoder, accuracy, X_test, y_test, y_pred

classifier, scaler, label_encoder, model_accuracy, X_test, y_test, y_pred = build_weather_model()

# Show model performance
st.subheader("Model Performance")
st.write(f"✅ Model trained with accuracy: {model_accuracy:.2%}")

# Inferential Statistics Section
st.header("📊 Inferential Statistics")

# Tabs for different statistical analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Descriptive Stats", 
    "Hypothesis Testing", 
    "Feature Analysis", 
    "Confusion Matrix"
])

with tab1:
    st.subheader("Descriptive Statistics")
    st.write(weather_data[['temp_max', 'temp_min', 'precipitation']].describe())
    
    # Weather distribution
    st.subheader("Weather Type Distribution")
    weather_counts = weather_data['weather'].value_counts()
    fig, ax = plt.subplots()
    weather_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Frequency of Weather Types")
    ax.set_xlabel("Weather Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with tab2:
    st.subheader("Hypothesis Testing")
    
    # T-test for temperature differences between rain and sun
    sun_temp = weather_data[weather_data['weather'] == 'sun']['temp_max']
    rain_temp = weather_data[weather_data['weather'] == 'rain']['temp_max']
    
    t_stat, p_value = stats.ttest_ind(sun_temp, rain_temp, equal_var=False)
    
    st.write("**Independent T-test:** Sun vs Rain maximum temperatures")
    st.write(f"- T-statistic: {t_stat:.2f}")
    st.write(f"- P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("The difference in maximum temperatures between sunny and rainy days is statistically significant (p < 0.05)")
    else:
        st.warning("No statistically significant difference in maximum temperatures between sunny and rainy days (p ≥ 0.05)")
    
    # ANOVA for multiple weather types
    st.write("\n**ANOVA:** Comparing precipitation across all weather types")
    weather_groups = [weather_data[weather_data['weather'] == weather_type]['precipitation'] for weather_type in weather_data['weather'].unique()]
    f_stat, p_value_anova = stats.f_oneway(*weather_groups)
    
    st.write(f"- F-statistic: {f_stat:.2f}")
    st.write(f"- P-value: {p_value_anova:.4f}")
    
    if p_value_anova < 0.05:
        st.success("There are significant differences in precipitation amounts between weather types (p < 0.05)")
    else:
        st.warning("No significant differences in precipitation amounts between weather types (p ≥ 0.05)")

with tab3:
    st.subheader("Feature Importance Analysis")
    
    # Feature importance from Random Forest
    feature_importance = classifier.feature_importances_
    feature_names = ['temp_max', 'temp_min', 'precipitation']
    
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=feature_names, ax=ax, palette="viridis")
    ax.set_title("Feature Importance from Random Forest")
    st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Matrix")
    numeric_data = weather_data[['temp_max', 'temp_min', 'precipitation']]
    correlation_matrix = numeric_data.corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Between Features")
    st.pyplot(fig)

with tab4:
    st.subheader("Confusion Matrix")
    
    # Create confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    weather_classes = label_encoder.classes_
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=weather_classes, yticklabels=weather_classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=weather_classes, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

# Prediction form
st.header("🌤️ Make a Prediction")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        max_temp = st.number_input("Max temperature (°C)", 
                                  min_value=-20.0, max_value=50.0, value=15.0, step=1.0)
        min_temp = st.number_input("Min temperature (°C)", 
                                  min_value=-20.0, max_value=50.0, value=8.0)
    
    with col2:
        precipitation = st.number_input("Precipitation (inches)", 
                                        min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    
    submitted = st.form_submit_button("Predict Weather")
    
    if submitted:
        # Prepare input
        input_data = [[max_temp, min_temp, precipitation]]
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        predicted_weather = classifier.predict(input_scaled)[0]
        weather_condition = label_encoder.inverse_transform([predicted_weather])[0]
        
        # Get prediction probabilities
        probabilities = classifier.predict_proba(input_scaled)[0]
        
        # Display result
        st.success(f"🌈 Predicted Weather: **{weather_condition}**")
        
        # Show prediction probabilities
        st.subheader("Prediction Probabilities")
        prob_data = pd.DataFrame({
            'Weather Type': label_encoder.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Probability', y='Weather Type', data=prob_data, ax=ax, palette='viridis')
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)
        
        # Display weather-specific emoji
        if weather_condition == "sun":
            st.write("☀️ Sunny day ahead! Perfect for outdoor activities.")
        elif weather_condition == "rain":
            st.write("🌧️ Don't forget your umbrella!")
        elif weather_condition == "fog":
            st.write("🌫️ Visibility might be low, drive carefully.")
        elif weather_condition == "drizzle":
            st.write("🌦️ Light rain expected, a jacket might be useful.")
        elif weather_condition == "snow":
            st.write("❄️ Bundle up! Snow is coming.")

# Add some space at the bottom
st.markdown("---")
st.caption("Note: This model was trained on historical Seattle weather data.")
