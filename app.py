import streamlit as st
import numpy as np
import joblib
from predictor import predict_rating 

st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")

st.title("Restaurant Rating Predictor")
st.write("""This app predicts the rating of a restaurant based on its features. Fill in the form below and click 'Predict Rating' to get a prediction.""")

@st.cache_resource
def load_model():
    try:
        model_path = 'restaurant_rating_predictor.pkl'
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_artifacts = load_model()

if model_artifacts:
    st.success("Model loaded successfully!")

    model = model_artifacts['model']
    encoders = model_artifacts['encoders']
    mlb = model_artifacts['cuisine_mlb']
    selected_features = model_artifacts['selected_features']
    model_type = model_artifacts['model_type']
    feature_importance = model_artifacts['feature_importances']

    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"Model Type: {model_type}")
    st.sidebar.write(f"R² Score: {model_artifacts['performance']['r2']:.4f}")
    st.sidebar.write(f"RMSE: {model_artifacts['performance']['rmse']:.4f}")

    with st.sidebar:
        st.subheader("Top 10 Most Important Features")
        top_features = feature_importance.head(10)
        for i, row in top_features.iterrows():
            st.write(f"{row['Feature']} (Importance: {row['Importance']:.4f})")

    with st.form("restaurant_form"):
        st.subheader("Restaurant Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            restaurant_name = st.text_input("Restaurant Name")
            city = st.selectbox("City", sorted(list(encoders['City'].classes_)))
            locality = st.selectbox("Locality", sorted(list(encoders['Locality'].classes_))[:100])  
            currency = st.selectbox("Currency", sorted(list(encoders['Currency'].classes_)))
            avg_cost = st.number_input("Average Cost for Two", min_value=0, value=500)
            price_range = st.slider("Price Range", 1, 4, 2)
            
        with col2:
            all_cuisines = sorted(list(mlb.classes_))
            default_cuisines = ['Italian', 'American'] if 'Italian' in all_cuisines and 'American' in all_cuisines else all_cuisines[:2]
            cuisines = st.multiselect("Cuisines", all_cuisines, default=default_cuisines)
            has_table_booking = st.selectbox("Has Table Booking", ['Yes', 'No'])
            has_online_delivery = st.selectbox("Has Online Delivery", ['Yes', 'No'])
            is_delivering_now = st.selectbox("Is Delivering Now", ['Yes', 'No'])
            switch_to_order_menu = st.selectbox("Switch to Order Menu", ['Yes', 'No'])
            votes = st.number_input("Votes", min_value=0, value=100)
            country_code = st.number_input("Country Code", min_value=1, value=1)
        
        cuisine_count = len(cuisines)
        log_cost = np.log1p(avg_cost)

        submit_button = st.form_submit_button("Predict Rating")

    if submit_button:
        try:
            cuisines_str = ", ".join(cuisines)
            
            restaurant_data = {
                'City': city,
                'Locality': locality,
                'Locality Verbose': locality, 
                'Currency': currency,
                'Has Table booking': has_table_booking,
                'Has Online delivery': has_online_delivery,
                'Is delivering now': is_delivering_now,
                'Switch to order menu': switch_to_order_menu,
                'Cuisines': cuisines_str,
                'Average Cost for two': avg_cost,
                'Price range': price_range,
                'Votes': votes,
                'Country Code': country_code,
                'Cuisine_Count': cuisine_count,
                'Log_Cost': log_cost
            }

            predicted_rating = predict_rating(restaurant_data, model_artifacts)

            final_prediction = predict_rating(restaurant_data, model_artifacts)

            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Rating", f"{final_prediction:.2f}/5.0")
            
            with col2:
                if final_prediction >= 4.5:
                    rating_text = "Excellent"
                elif final_prediction >= 4.0:
                    rating_text = "Very Good"
                elif final_prediction >= 3.5:
                    rating_text = "Good"
                elif final_prediction >= 3.0:
                    rating_text = "Average"
                elif final_prediction >= 2.0:
                    rating_text = "Poor"
                else:
                    rating_text = "Not Recommended"
                
                st.metric("Rating Category", rating_text)
            
            with col3:
                if final_prediction >= 4.5:
                    color = "Dark Green"
                elif final_prediction >= 4.0:
                    color = "Green"
                elif final_prediction >= 3.5:
                    color = "Yellow"
                elif final_prediction >= 3.0:
                    color = "Orange"
                else:
                    color = "Red"
                
                st.metric("Rating Color", color)

            st.subheader("Restaurant Summary")
            summary = f"""
            **{restaurant_name}** is located in {locality}, {city}. 
            It serves {cuisines_str} cuisine with a price range of {price_range}/4 
            (average cost for two: {currency} {avg_cost}). 
            """
            st.markdown(summary)
            
            st.subheader("What Influenced This Rating")
            
            factors = []
            if cuisine_count >= 3:
                factors.append("Diverse cuisine options")
            if votes > 100:
                factors.append("High number of votes")
            if has_online_delivery == "Yes":
                factors.append("Online delivery availability")
            if has_table_booking == "Yes":
                factors.append("Table booking service")
            if price_range >= 3:
                factors.append("Premium pricing")
            elif price_range <= 2:
                factors.append("Affordable pricing")
            
            for factor in factors:
                st.write(f"✓ {factor}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error(f"Error details: {str(e)}")
else:
    st.error("Failed to load model. Please make sure the model file exists and is valid.")