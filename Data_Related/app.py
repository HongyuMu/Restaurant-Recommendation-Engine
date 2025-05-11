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

file_id = "1wGPVdSei3NHKaZ0rHVILTvBOHAK2MVMJ"
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
st.caption("""
💲 **Price Range Meaning**  
- 1: Under $10 per person  
- 2: $11–30 per person  
- 3: $31–60 per person  
- 4: Over $61 per person
""")

st.subheader("📅 Contextual Preferences")

selected_day = st.selectbox("Day of the Week You Plan to Visit:", 
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

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

def review_confidence_weight(n_reviews, min_reviews=20, threshold=400):
    if n_reviews < min_reviews:
        return 0.5  # penalty for very low review count
    elif n_reviews >= threshold:
        return 1.0  # full trust
    else:
        # Smooth linear scale from penalty to trust
        return 0.5 + 0.5 * (n_reviews - min_reviews) / (threshold - min_reviews)


# --- Scoring Function ---
def compute_restaurant_scores(df, weight_dict, selected_features, review_boost=0, penalty_strength = 0.15):
    features = selected_features.copy()
    max_review_weight = 0.15
    actual_review_weight = review_boost * max_review_weight

    # Optional log feature
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
    base_scores = scaled @ weights
    df['base_score'] = base_scores

    scaler_var = MinMaxScaler()
    df[['sentiment_var_norm', 'star_var_norm']] = scaler_var.fit_transform(
        df[['sentiment_variability', 'star_variability']]
    )
    df['consistency_penalty'] = 1 - penalty_strength * (df['sentiment_var_norm'] + df['star_var_norm']) / 2
    df['penalized_score'] = df['base_score'] * df['consistency_penalty']

    df['confidence_weight'] = df['review_count'].apply(review_confidence_weight)

    # Normalize score by dividing the best performer
    df['final_score'] = 10 * (df['penalized_score'] * df['confidence_weight']) / \
                        (df['penalized_score'] * df['confidence_weight']).max()

    return df

# --- User Input: Feature Weights ---
st.subheader("Set Your Preferences")
food_w = st.slider("Food Quality", 0.0, 1.0, 0.2)
service_w = st.slider("Service Quality", 0.0, 1.0, 0.2)
ambience_w = st.slider("Resturant Ambience", 0.0, 1.0, 0.2)
value_w = st.slider("Portion Size", 0.0, 1.0, 0.2)
stars_w = st.slider("Overall Star", 0.0, 1.0, 0.2)
review_pref = st.slider("Total Review Count", 0.0, 1.0, 0.3)

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

        weekday_cols = ['Monday_star', 'Tuesday_star', 'Wednesday_star', 'Thursday_star', 'Friday_star']
        weekend_cols = ['Saturday_star', 'Sunday_star']
        scored_df['weekday_avg_star'] = scored_df[weekday_cols].mean(axis=1)
        scored_df['weekend_avg_star'] = scored_df[weekend_cols].mean(axis=1)

        top_n = 10
        top_restaurants = scored_df.sort_values(by='final_score', ascending=False).head(top_n)

        st.subheader(f"Top {top_n} Recommended Restaurants")

        for _, row in top_restaurants.iterrows():
            # --- Determine contextual rating based on selected day ---
            if selected_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                contextual_rating = f"{row['weekday_avg_star']:.2f} (weekday avg)"
            else:
                contextual_rating = f"{row['weekend_avg_star']:.2f} (weekend avg)"

            col1, col2 = st.columns([2, 3])  # left: basic info; right: details

            with col1:
                yelp_link = f"https://www.yelp.com/biz/{row['business_id']}"
                st.markdown(f"### [{row['name']}]({yelp_link})")
                st.markdown(f"⭐ **{row.get('stars', 'N/A')} stars** ({row.get('review_count', 0)} reviews)")
                st.markdown(f"🏷️ **Categories**: {row.get('categories', 'N/A')}")
                st.markdown(f"📍 `{row.get('address', 'N/A')}`")
                st.markdown(f"💲 Price Level: `{row.get('RestaurantsPriceRange2', 'N/A')}`")
                st.markdown(f"🗓️ **Contextual Rating**: `{contextual_rating}`")

            with col2:
                st.markdown("#### 📊 Summary")
                st.markdown(f"""
                - 🍽️ **Food Quality Score**: `{row.get('food_sentiment', 'N/A'):.3f}`
                - 🙋 **Service Score**: `{row.get('service_sentiment', 'N/A'):.3f}`
                - 🪑 **Ambience Score**: `{row.get('ambience_sentiment', 'N/A'):.3f}`
                - 💰 **Value Score**: `{row.get('value_sentiment', 'N/A'):.3f}`
                - 📉 **Star Variability**: `{row.get('star_variability', 'N/A'):.3f}`
                - 🎯 **Final Score**: `{row['final_score']:.2f}` out of 10
                """)

            st.markdown("---")  # divider between cards

