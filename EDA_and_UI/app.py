import streamlit as st
import pandas as pd
import numpy as np
import re
import datetime
import ast
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

file_id = "1dqYsDQVmc4rf0LH7TiiUOKuAQEsXCSJv"
url = f"https://drive.google.com/uc?id={file_id}"
business_df = load_data(url)
st.title("Restaurant Recommender")
st.markdown("Select your preferences to get personalized restaurant recommendations.")

# --- User Input: Cuisine ---
available_cuisines = sorted([
    'Afghan', 'African', 'American', 'Argentine', 'Asian Fusion', 'Barbeque',
    'Brazilian', 'Breakfast & Brunch', 'British', 'Buffets', 'Burgers', 'Cafes',
    'Cajun', 'Caribbean', 'Chinese', 'Comfort Food', 'Creole', 'Cuban', 'Delis',
    'Diners', 'Ethiopian', 'Fast Food', 'Filipino', 'French', 'Gastropubs',
    'German', 'Greek', 'Halal', 'Hawaiian', 'Indian', 'Irish', 'Italian',
    'Japanese', 'Korean', 'Latin American', 'Mediterranean', 'Mexican',
    'Middle Eastern', 'Nightlife', 'Noodles', 'Pakistani', 'Peruvian', 'Pizza', 'Polish',
    'Portuguese', 'Ramen', 'Russian', 'Sandwiches', 'Seafood', 'Soul Food',
    'Soup', 'Southern', 'Spanish', 'Steakhouses', 'Sushi Bars', 'Taiwanese',
    'Tapas', 'Tex-Mex', 'Thai', 'Turkish', 'Vegan', 'Vegetarian', 'Vietnamese', 'Wine'
])
cuisine_options = ['No specific preference'] + available_cuisines
selected_cuisines = st.multiselect("Select your preferred cuisine(s):", cuisine_options, default=['No specific preference'])

# --- User Input: Price Level ---
price_options = [1, 2, 3, 4]
selected_prices = st.multiselect("Select acceptable price levels:", price_options, default=price_options)

# --- User Input: Open Now? ---
show_open_only = st.checkbox("Show only restaurants open now")

def parse_time_str(time_str):
    """Converts 'HH:MM' or 'H:M' to float hour (e.g., '14:30' → 14.5)"""
    parts = time_str.strip().split(':')
    if len(parts) == 2:
        hour, minute = int(parts[0]), int(parts[1])
        return hour + minute / 60
    return float(time_str)  # fallback if already float-like

def is_open_now(row):
    now = datetime.datetime.now()
    today = now.strftime('%A')
    current_time = now.hour + now.minute / 60

    try:
        days = ast.literal_eval(row['open_days'])
        times = ast.literal_eval(row['time_span'])
    except (ValueError, SyntaxError):
        return False

    for d, t in zip(days, times):
        if d == today:
            try:
                start_str, end_str = t.split('-')
                start = parse_time_str(start_str)
                end = parse_time_str(end_str)
                if start <= current_time <= end:
                    return True
            except Exception:
                continue
    return False

# --- Scoring Function ---
def compute_restaurant_scores(df, weight_dict, selected_features, review_boost=0):
    features = selected_features.copy()
    max_review_weight = 0.1
    actual_review_weight = review_boost * max_review_weight

    if actual_review_weight > 0:
        df['log_review_count'] = np.log1p(df['review_count'])
        features.append('log_review_count')

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    weights = []

    for feat in features:
        weights.append(actual_review_weight if feat == 'log_review_count' else weight_dict.get(feat, 0))

    weights = np.array(weights)
    weights /= weights.sum()
    df['final_score'] = 10 * (scaled @ weights) / np.max(scaled @ weights)
    return df

# --- User Input: Feature Weights ---
st.subheader("Set Your Preferences")
food_w = st.slider("Food Sentiment Weight", 0.0, 1.0, 0.2)
service_w = st.slider("Service Sentiment Weight", 0.0, 1.0, 0.2)
ambience_w = st.slider("Ambience Sentiment Weight", 0.0, 1.0, 0.2)
value_w = st.slider("Value Sentiment Weight", 0.0, 1.0, 0.2)
stars_w = st.slider("Stars Weight", 0.0, 1.0, 0.2)
review_pref = st.slider("Emphasis on Review Volume", 0.0, 1.0, 0.3)

# --- Button to Trigger Search ---
if st.button("🔍 Begin Searching for Restaurants"):

    # --- Apply Cuisine Filter ---
    if 'No specific preference' in selected_cuisines:
        filtered_df = business_df.copy()
    else:
        pattern = '|'.join([re.escape(c) for c in selected_cuisines])
        filtered_df = business_df[business_df['categories'].str.contains(pattern, case=False, na=False)]

    # --- Apply Price Filter ---
    filtered_df = filtered_df[filtered_df['RestaurantsPriceRange2'].isin(selected_prices)]
    
    if show_open_only:
        filtered_df['is_open_now'] = filtered_df.apply(is_open_now, axis=1)
        filtered_df = filtered_df[filtered_df['is_open_now']]

    total = food_w + service_w + ambience_w + value_w + stars_w
    weights_dict = {
        'food_sentiment': food_w / total,
        'service_sentiment': service_w / total,
        'ambience_sentiment': ambience_w / total,
        'value_sentiment': value_w / total,
        'stars': stars_w / total
    }

    if filtered_df.empty:
        st.warning("Oops! There are no restaurants meeting these criteria at this moment. Please broaden your scope.")
    else:
        scored_df = compute_restaurant_scores(
            filtered_df.copy(),
            weight_dict=weights_dict,
            selected_features=['food_sentiment', 'service_sentiment', 'ambience_sentiment', 'value_sentiment', 'stars'],
            review_boost=review_pref
        )

        top_n = 10
        top_restaurants = scored_df.sort_values(by='final_score', ascending=False).head(top_n)

        st.subheader(f"Top {top_n} Recommended Restaurants")
        for _, row in top_restaurants.iterrows():
            st.markdown(f"""
            ### {row['name']}
            - Stars: {row.get('stars', 'N/A')}
            - Reviews: {row.get('review_count', 'N/A')}
            - Categories: {row.get('categories', 'N/A')}
            - Address: {row.get('address', 'N/A')}
            - Open Now: {"✅ Yes" if row.get('is_open_now') else "❌ No"}
            - Score: `{row['final_score']:.2f}`
            """)
