import streamlit as st
import pandas as pd
import re
import datetime
import ast

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

file_id = "1dqYsDQVmc4rf0LH7TiiUOKuAQEsXCSJv"
url = f"https://drive.google.com/uc?id={file_id}"
business_df = load_data(url)
st.title("Restaurant Recommender")
st.markdown("Select your preferences to get personalized restaurant recommendations.")

# Extract unique cuisines and sort alphabetically
available_cuisines = sorted([
    'Afghan', 'African', 'American', 'Argentine', 'Asian Fusion', 'Barbeque',
    'Brazilian', 'Breakfast & Brunch', 'British', 'Buffets', 'Burgers', 'Cafes',
    'Cajun/Creole', 'Caribbean', 'Chinese', 'Comfort Food', 'Cuban', 'Delis',
    'Diners', 'Ethiopian', 'Fast Food', 'Filipino', 'French', 'Gastropubs',
    'German', 'Greek', 'Halal', 'Hawaiian', 'Indian', 'Irish', 'Italian',
    'Japanese', 'Korean', 'Latin American', 'Mediterranean', 'Mexican',
    'Middle Eastern', 'Noodles', 'Pakistani', 'Peruvian', 'Pizza', 'Polish',
    'Portuguese', 'Ramen', 'Russian', 'Sandwiches', 'Seafood', 'Soul Food',
    'Soup', 'Southern', 'Spanish', 'Steakhouses', 'Sushi Bars', 'Taiwanese',
    'Tex-Mex', 'Thai', 'Turkish', 'Vegan', 'Vegetarian', 'Vietnamese'
])

cuisine_options = ['No specific preference'] + available_cuisines
selected_cuisines = st.multiselect("Select your preferred cuisine(s):", cuisine_options, default=['No specific preference'])

# Filter based on cuisine preference
if 'No specific preference' in selected_cuisines:
    filtered_df = business_df
else:
    # regex for multiple options, also prevent regex sensitive characters in business names
    pattern = '|'.join([re.escape(cuisine) for cuisine in selected_cuisines])
    filtered_df = business_df[business_df['categories'].str.contains(pattern, case=False, na=False)]


# Add multi-select for price level
price_options = [1, 2, 3, 4]
selected_prices = st.multiselect(
    "Select acceptable price levels (can choose multiple):", price_options, default=price_options
)
# Apply price filter (assuming the column is named 'price')
filtered_df = filtered_df[filtered_df['RestaurantsPriceRange2'].isin(selected_prices)]

def is_open_now(row):
    # Get current day and time
    now = datetime.datetime.now()
    today = now.strftime('%A')  # e.g., 'Monday'
    current_time = now.hour + now.minute / 60  # convert to decimal hours

    # Convert string to actual lists
    try:
        days = ast.literal_eval(row['open_days'])
        times = ast.literal_eval(row['time_span'])
    except (ValueError, SyntaxError):
        return False  # skip invalid entries

    # Match today's time span
    for d, t in zip(days, times):
        if d == today:
            try:
                start_str, end_str = t.split('-')
                start = float(start_str)
                end = float(end_str)
                if start <= current_time <= end:
                    return True
            except:
                continue

    return False

business_df['is_open_now'] = business_df.apply(is_open_now, axis=1)
open_now_df = business_df[business_df['is_open_now']]

if st.checkbox("Show only restaurants open now"):
    display_df = open_now_df
else:
    display_df = business_df

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