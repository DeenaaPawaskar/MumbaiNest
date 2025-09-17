import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ======================
# PAGE SETUP
# ======================
st.set_page_config(
    page_title="🏠 Mumbai House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .price-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .future-price-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: #333;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .accuracy-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .input-header {
        background: linear-gradient(90deg, #4ECDC4, #44A08D);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# TITLE AND INTRODUCTION
# ======================
st.markdown('<div class="main-title">🏠 Mumbai House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Mumbai property prices using real market data! 🚀</div>', unsafe_allow_html=True)

# ======================
# LOAD KAGGLE DATASET
# ======================
@st.cache_data
def load_mumbai_data():
    """Load Mumbai house price dataset from Kaggle"""
    try:
        # Try to load with kagglehub
        import kagglehub
        path = kagglehub.dataset_download("kevinnadar22/mumbai-house-price-data-70k-entries")

        # Look for CSV files in the downloaded path
        import os
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

        if csv_files:
            df = pd.read_csv(os.path.join(path, csv_files[0]))
        else:
            raise FileNotFoundError("No CSV file found")

    except Exception as e:
        st.warning(f"⚠️ Could not load Kaggle dataset: {str(e)}")
        st.info("Creating sample Mumbai data for demonstration...")

        # Create realistic Mumbai data as backup
        np.random.seed(42)
        n_samples = 5000

        mumbai_areas = [
            'Bandra West', 'Andheri West', 'Juhu', 'Powai', 'Worli', 'Lower Parel',
            'Colaba', 'Chembur', 'Thane West', 'Malad West', 'Borivali West',
            'Kandivali West', 'Santacruz West', 'Vile Parle', 'Khar West',
            'Versova', 'Oshiwara', 'Lokhandwala', 'Mahim', 'Dadar West'
        ]

        # Generate realistic data
        df = pd.DataFrame({
            'area': np.random.normal(900, 400, n_samples),
            'bedroom': np.random.randint(1, 5, n_samples),
            'bathroom': np.random.randint(1, 4, n_samples),
            'locality': np.random.choice(mumbai_areas, n_samples),
            'price': np.random.normal(8000000, 3000000, n_samples)  # Price in rupees
        })

        # Ensure positive values
        df['area'] = np.maximum(df['area'], 300)
        df['price'] = np.maximum(df['price'], 2000000)  # Minimum 20 lakhs

        # Make prices more realistic based on area and location
        area_factor = df['area'] * np.random.uniform(8000, 15000, n_samples)
        bedroom_factor = df['bedroom'] * np.random.uniform(200000, 500000, n_samples)
        df['price'] = area_factor + bedroom_factor + np.random.normal(1000000, 500000, n_samples)
        df['price'] = np.maximum(df['price'], 1500000)  # Minimum 15 lakhs

    # Clean the data
    df = df.dropna()

    # Standardize column names
    column_mapping = {
        'bedroom': 'bedrooms',
        'bathroom': 'bathrooms',
        'bedroom_num': 'bedrooms',
        'bathroom_num': 'bathrooms'
    }
    df = df.rename(columns=column_mapping)

    # Ensure we have required columns
    required_columns = ['area', 'bedrooms', 'bathrooms', 'locality', 'price']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        return None

    # Filter realistic values
    df = df[(df['area'] > 200) & (df['area'] < 10000)]  # 200-10000 sq ft
    df = df[(df['bedrooms'] >= 1) & (df['bedrooms'] <= 6)]
    df = df[(df['bathrooms'] >= 1) & (df['bathrooms'] <= 5)]
    df = df[df['price'] > 1000000]  # Minimum 10 lakhs

    return df

# Load the dataset
df = load_mumbai_data()

if df is None:
    st.stop()

# Display dataset info
st.success(f"✅ Loaded {len(df):,} Mumbai property records!")

# Show sample data
with st.expander("📊 View Sample Data"):
    st.dataframe(df.head())
    st.write(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

# ======================
# TRAIN THE MODEL
# ======================
@st.cache_data
def train_mumbai_model(data):
    """Train linear regression model on Mumbai data"""

    # Prepare features
    X = data[['area', 'bedrooms', 'bathrooms']].copy()

    # Encode locality
    le = LabelEncoder()
    X['locality_encoded'] = le.fit_transform(data['locality'])

    y = data['price']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate performance metrics
    y_pred = model.predict(X_test)

    # Simple accuracy: predictions within 15% of actual price
    accuracy_mask = np.abs(y_test - y_pred) / y_test < 0.15
    accuracy_percentage = accuracy_mask.mean() * 100

    # R² score for model quality
    r2 = r2_score(y_test, y_pred)

    return model, le, accuracy_percentage, r2, X.columns

# Train the model
model, locality_encoder, accuracy, r2_score_val, feature_columns = train_mumbai_model(df)

# ======================
# SIDEBAR - USER INPUT
# ======================
st.sidebar.markdown('<div class="input-header"><h2>🏡 Enter Your Property Details</h2></div>', unsafe_allow_html=True)

# Get unique localities from dataset
available_localities = sorted(df['locality'].unique())

# User inputs
user_area = st.sidebar.slider(
    "🏠 **Carpet Area (Square Feet)**",
    min_value=int(df['area'].min()),
    max_value=int(df['area'].max()),
    value=int(df['area'].median()),
    step=50,
    help="Total carpet area of your property"
)

user_bedrooms = st.sidebar.selectbox(
    "🛏️ **Number of Bedrooms**",
    options=sorted(df['bedrooms'].unique()),
    index=1,
    help="Total bedrooms in your property"
)

user_bathrooms = st.sidebar.selectbox(
    "🚿 **Number of Bathrooms**",
    options=sorted(df['bathrooms'].unique()),
    index=0,
    help="Total bathrooms in your property"
)

user_locality = st.sidebar.selectbox(
    "📍 **Locality in Mumbai**",
    options=available_localities,
    help="Choose your area in Mumbai"
)

st.sidebar.markdown("---")

# Future prediction inputs
st.sidebar.markdown("### 📈 **Future Value Calculator**")
years_ahead = st.sidebar.slider(
    "**Predict Price After (Years)**",
    min_value=1,
    max_value=15,
    value=5,
    help="How many years from now do you want to predict?"
)

growth_rate = st.sidebar.slider(
    "**Expected Annual Growth Rate (%)**",
    min_value=2.0,
    max_value=15.0,
    value=7.0,
    step=0.5,
    help="Expected yearly appreciation in Mumbai real estate"
) / 100

# ======================
# MAKE PREDICTION FUNCTION
# ======================
def predict_property_price(area, bedrooms, bathrooms, locality):
    """Predict property price based on user inputs"""
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'locality_encoded': [locality_encoder.transform([locality])[0]]
        })

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        # Ensure reasonable price (minimum 10 lakhs)
        return max(predicted_price, 1000000)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 5000000  # Default 50 lakhs

# ======================
# GET PREDICTIONS
# ======================
current_price = predict_property_price(user_area, user_bedrooms, user_bathrooms, user_locality)
future_price = current_price * ((1 + growth_rate) ** years_ahead)

# ======================
# MAIN DASHBOARD DISPLAY
# ======================
col1, col2 = st.columns([2, 1])

with col1:
    # Current Price Display
    st.markdown("## 💰 **Your Property Price Today**")
    st.markdown(f"""
    <div class="price-box">
        <h1 style="font-size: 3rem; margin: 0;">₹{current_price/100000:.1f} Lakhs</h1>
        <p style="font-size: 1.2rem; margin: 5px 0;">Current Market Value</p>
        <p style="font-size: 1rem; opacity: 0.9;">₹{current_price/user_area:,.0f} per sq ft</p>
    </div>
    """, unsafe_allow_html=True)

    # Future Price Display
    st.markdown("## 🚀 **Future Value Projection**")
    growth_amount = future_price - current_price
    st.markdown(f"""
    <div class="future-price-box">
        <h2 style="margin: 0;">In {years_ahead} years, your property could be worth:</h2>
        <h1 style="font-size: 2.5rem; margin: 10px 0; color: #E67E22;">₹{future_price/100000:.1f} Lakhs</h1>
        <p style="font-size: 1.1rem; margin: 0;">
            📈 That's <strong>₹{growth_amount/100000:.1f} Lakhs more</strong> than today!<br>
            💡 Total growth: <strong>{((future_price/current_price - 1)*100):.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Model Performance
    st.markdown("## 🎯 **Prediction Accuracy**")
    st.markdown(f"""
    <div class="accuracy-box">
        <h2 style="color: #27AE60; font-size: 2.5rem; margin: 0;">{accuracy:.0f}%</h2>
        <p style="margin: 5px 0; font-size: 1.1rem;"><strong>Accuracy Rate</strong></p>
        <p style="margin: 0; font-size: 0.9rem;">Predictions accurate within 15% range</p>
    </div>
    """, unsafe_allow_html=True)

    # Property Summary
    st.markdown("### 📊 **Your Property Summary**")
    st.info(f"""
    **🏠 Area:** {user_area:,} sq ft  
    **🛏️ Bedrooms:** {user_bedrooms}  
    **🚿 Bathrooms:** {user_bathrooms}  
    **📍 Location:** {user_locality}  
    **💰 Rate:** ₹{current_price/user_area:,.0f}/sq ft
    """)

# ======================
# LOCALITY COMPARISON
# ======================
st.markdown("---")
st.markdown("## 🏘️ **How Does Your Area Compare?**")

# Calculate average prices by locality
locality_avg = df.groupby('locality')['price'].mean().sort_values(ascending=False)

# Create comparison chart highlighting user's locality
colors = ['#FF6B6B' if loc == user_locality else '#4ECDC4' for loc in locality_avg.index]

fig_locality = go.Figure()
fig_locality.add_trace(go.Bar(
    x=locality_avg.values / 100000,
    y=locality_avg.index,
    orientation='h',
    marker_color=colors,
    text=[f'₹{val/100000:.1f}L' for val in locality_avg.values],
    textposition='auto',
))

fig_locality.update_layout(
    title=f"Average Property Prices by Locality (Your area: {user_locality} highlighted in red)",
    xaxis_title="Average Price (₹ Lakhs)",
    yaxis_title="Mumbai Localities",
    height=600,
    showlegend=False
)

st.plotly_chart(fig_locality, use_container_width=True)

# ======================
# FEATURE IMPORTANCE
# ======================
st.markdown("## 📊 **What Affects Your Property Price?**")

# Get actual feature importance from the model
feature_names = ['Area (sq ft)', 'Bedrooms', 'Bathrooms', 'Locality']
feature_importance = np.abs(model.coef_)
importance_normalized = (feature_importance / feature_importance.sum()) * 100

importance_df = pd.DataFrame({
    'Factor': feature_names,
    'Importance': importance_normalized
}).sort_values('Importance', ascending=True)

fig_importance = px.bar(
    importance_df,
    x='Importance',
    y='Factor',
    orientation='h',
    title="Which factors affect your property price the most?",
    color='Importance',
    color_continuous_scale='viridis',
    text='Importance'
)

fig_importance.update_traces(
    texttemplate='%{text:.1f}%',
    textposition='inside'
)
fig_importance.update_layout(
    height=400,
    showlegend=False,
    xaxis_title="Impact on Price (%)"
)

st.plotly_chart(fig_importance, use_container_width=True)

# ======================
# GROWTH PROJECTION TIMELINE
# ======================
st.markdown("## 💡 **Investment Growth Timeline**")

col3, col4 = st.columns([2, 1])

with col3:
    # Year-by-year projection
    years = list(range(0, years_ahead + 1))
    projected_values = [current_price * ((1 + growth_rate) ** year) for year in years]

    growth_df = pd.DataFrame({
        'Year': years,
        'Value': [v/100000 for v in projected_values]
    })

    fig_growth = px.line(
        growth_df,
        x='Year',
        y='Value',
        title=f'Property Value Growth Projection ({growth_rate*100:.1f}% annually)',
        markers=True,
        line_shape='spline'
    )

    fig_growth.update_traces(
        line_color='#FF6B6B',
        marker_size=8,
        line_width=3
    )

    fig_growth.update_layout(
        xaxis_title="Years from Now",
        yaxis_title="Property Value (₹ Lakhs)",
        height=400
    )

    st.plotly_chart(fig_growth, use_container_width=True)

with col4:
    st.markdown("### 💰 **Investment Summary**")

    total_growth = future_price - current_price
    annual_growth = total_growth / years_ahead
    roi_percentage = ((future_price / current_price) - 1) * 100

    st.markdown(f"""
    <div class="price-box">
        <h4>📈 Investment Analysis:</h4>
        <p><strong>Current Value:</strong> ₹{current_price/100000:.1f}L</p>
        <p><strong>Future Value:</strong> ₹{future_price/100000:.1f}L</p>
        <p><strong>Total Gain:</strong> ₹{total_growth/100000:.1f}L</p>
        <p><strong>Annual Gain:</strong> ₹{annual_growth/100000:.1f}L</p>
        <p><strong>Total ROI:</strong> {roi_percentage:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ======================
# AREA DISTRIBUTION
# ======================
st.markdown("## 📈 **Mumbai Property Market Insights**")

col5, col6 = st.columns(2)

with col5:
    # Price distribution
    fig_dist = px.histogram(
        df,
        x=df['price']/100000,
        bins=30,
        title="Mumbai Property Price Distribution",
        labels={'x': 'Price (₹ Lakhs)', 'y': 'Number of Properties'}
    )

    # Add line for user's property price
    fig_dist.add_vline(
        x=current_price/100000,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Your Property: ₹{current_price/100000:.1f}L"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

with col6:
    # Area vs Price scatter
    fig_scatter = px.scatter(
        df.sample(1000),
        x='area',
        y=df.sample(1000)['price']/100000,
        title="Area vs Price Relationship",
        labels={'x': 'Area (sq ft)', 'y': 'Price (₹ Lakhs)'},
        opacity=0.6
    )

    # Add user's property point
    fig_scatter.add_scatter(
        x=[user_area],
        y=[current_price/100000],
        mode='markers',
        marker=dict(size=15, color='red'),
        name='Your Property'
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.markdown("## 💡 **Mumbai Real Estate Tips**")

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    ### 🏠 **For Property Buyers:**
    - **Western suburbs** typically command premium prices
    - **Connectivity** to business districts affects pricing significantly  
    - **Future infrastructure** projects boost long-term value
    - **Monsoon accessibility** impacts property desirability
    """)

with tips_col2:
    st.markdown("""
    ### 💰 **For Property Sellers:**
    - **Market timing** matters - avoid monsoon season
    - **Property staging** can increase perceived value
    - **Locality comparison** helps competitive pricing
    - **Growth projections** attract investment buyers
    """)

# Model details
st.markdown("---")
st.markdown("### 🔧 **Model Details**")
st.info(f"""
**Model Type:** Linear Regression  
**Training Data:** {len(df):,} Mumbai properties  
**Accuracy:** {accuracy:.1f}% (predictions within 15% of actual price)  
**R² Score:** {r2_score_val:.3f} (model explains {r2_score_val*100:.1f}% of price variation)
""")

st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
    <h4>🏠 Mumbai House Price Predictor</h4>
    <p>Powered by real Mumbai property data from Kaggle 📊</p>
    <p style='font-size: 0.8rem;'>Built with ❤️ using Streamlit & Linear Regression</p>
</div>
""", unsafe_allow_html=True)
