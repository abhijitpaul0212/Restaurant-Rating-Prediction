# streamlit_app.py

import streamlit as st
import pandas as pd

from src.RestaurantRatingPrediction.pipelines.prediction_pipeline import PredictPipeline
from src.RestaurantRatingPrediction.utils.utils import Utils
from src.RestaurantRatingPrediction.utils.data_processor import DBProcessor

# Ingest data from MongoDB
data = Utils().run_data_pipeline(DBProcessor(), "mongodb+srv://root:root@cluster0.k3s4vuf.mongodb.net/?retryWrites=true&w=majority", "zomato_database/reviews")

# Drop dolumns
data.drop(["rate", "_id"], axis=1, inplace=True)

# Design Streamlit Page
st.write("""
# Restaurant Rating Prediction
This app predicts the **Restaurant Rating**!
""")
st.write("---")

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header("Specify Input Parameters")


def user_input_features(data):
    order_online = st.sidebar.selectbox("Online Order:", ["Yes", "No"])
    book_table = st.sidebar.selectbox("Book Table:", ["Yes", "No"])
    location = st.sidebar.selectbox("Location:", ["others", "Banashankari", "Basavanagudi", "Jayanagar", "JP Nagar", "Bannerghatta Road", "BTM", "Electronic City", "HSR", 
                     "Marathahalli", "Sarjapur Road", "Shanti Nagar", "Koramangala 5th Block", "Richmond Road", "Koramangala 7th Block",
                     "Koramangala 4th Block", "Bellandur", "Whitefield", "Indiranagar", "Koramangala 1st Block", "Frazer Town", "MG Road", "Brigade Road",
                     "Lavelle Road", "Church Street", "Ulsoor", "Residency Road", "Malleshwaram", "Kammanahalli", "Koramangala 6th Block",
                     "Brookefield", "Rajajinagar", "Banaswadi", "Kalyan Nagar", "New BEL Road"])
    rest_type = st.sidebar.selectbox("Resturant Type", ["Casual Dining", "others", "Quick Bites", "Cafe", "Delivery", "Dessert Parlor", "Bakery", "Takeaway, Delivery","Casual Dining, Bar"])
    cuisines = st.sidebar.selectbox("Cuisines:", ["others", "South Indian, North Indian", "North Indian", "Cafe", "Bakery, Desserts", "Biryani", "South Indian", "North Indian, Chinese", "Ice Cream, Desserts", "Chinese",
                "Bakery", "Fast Food", "Mithai, Street Food", "Desserts", "South Indian, North Indian, Chinese", "Beverages", "Chinese, North Indian", "Desserts, Ice Cream", 
                "North Indian, Chinese, Biryani", "North Indian, South Indian", "North Indian, South Indian, Chinese"])
    cost_for_2 = st.number_input("Cost for 2:")
    type = st.sidebar.selectbox("Type", ["Buffet", "Cafes", "Delivery", "Desserts", "Dine-out", "Drinks & nightlife", "Pubs and bars"])
    votes = st.number_input("Votes:")

    data = {
                "online_order": order_online,
                "book_table": book_table,
                "location": location,
                "rest_type": rest_type,
                "cuisines": cuisines, 
                "cost_for_2": float(cost_for_2),
                "type": type,
                "votes": float(votes)
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features(data)

# Main Panel
# Print specified input parameters
st.header("Specified Input parameters")
st.write(df)
st.write("---")

predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)

st.header('Restaurant Rating Prediction Result')
st.write(round(prediction[0], 2))

st.write("---")
