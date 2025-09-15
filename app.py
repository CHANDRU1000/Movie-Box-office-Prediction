import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Movie Box Office Prediction", layout="centered")

# Load Linear Regression model & encoders
with open("linear_movie_model.pkl", "rb") as f:
    model, le_genre, le_month = pickle.load(f)

# App title
st.title("🍿🎬 Movie Box Office Prediction App")
st.write("Predict the box office revenue of a movie before release using **Linear Regression**!")

st.sidebar.header("Enter Movie Details:")

# Inputs
genre = st.sidebar.selectbox("Genre", le_genre.classes_)
budget = st.sidebar.number_input("Budget (in Crores)", min_value=1, max_value=1000, value=50)
cast_popularity = st.sidebar.slider("Cast Popularity (1-10)", 1.0, 10.0, 5.0)
director_popularity = st.sidebar.slider("Director Popularity (1-10)", 1.0, 10.0, 5.0)
marketing_spend = st.sidebar.number_input("Marketing Spend (in Crores)", min_value=0, max_value=500, value=20)
release_month = st.sidebar.selectbox("Release Month", le_month.classes_)

# Prediction button
if st.sidebar.button("Predict Revenue"):
    input_data = pd.DataFrame({
        "Genre": [le_genre.transform([genre])[0]],
        "Budget": [budget],
        "Cast_Popularity": [cast_popularity],
        "Director_Popularity": [director_popularity],
        "Marketing_Spend": [marketing_spend],
        "Release_Month": [le_month.transform([release_month])[0]],
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Box Office Collection: ₹ {prediction:,.2f} Crores")