import streamlit as st
import pandas as pd

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

file_path = "D:/360MoveData/Users/dell/Documents/GitHub/Restaurant-Recommendation-Engine/Data Related/business_with_sentiments.csv"
business_df = load_data(file_path)
st.title("Restaurant Recommender")
st.markdown("Select your preferences to get personalized restaurant recommendations.")

# Extract unique cuisines (simplify to main types first)
available_cuisines = ['Mexican', 'Chinese', 'Japanese', 'American', 'Italian', 'Thai']
cuisine_choice = st.selectbox("Choose a cuisine type:", available_cuisines)

filtered_df = business_df[business_df['categories'].str.contains(cuisine_choice, case=False, na=False)]

st.subheader("Set Your Preferences")

food_w = st.slider("Food Sentiment Weight", 0.0, 1.0, 0.2)
service_w = st.slider("Service Sentiment Weight", 0.0, 1.0, 0.2)
ambience_w = st.slider("Ambience Sentiment Weight", 0.0, 1.0, 0.2)
value_w = st.slider("Value Sentiment Weight", 0.0, 1.0, 0.2)
stars_w = st.slider("Stars Weight", 0.0, 1.0, 0.2)

# Normalize weights
total = food_w + service_w + ambience_w + value_w + stars_w
weights_dict = {
    'food_sentiment': food_w / total,
    'service_sentiment': service_w / total,
    'ambience_sentiment': ambience_w / total,
    'value_sentiment': value_w / total,
    'stars': stars_w / total
}

review_pref = st.slider("Emphasis on Review Volume", 0.0, 1.0, 0.3)