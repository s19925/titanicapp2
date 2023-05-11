import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def fill_empty_rows_with_medians(data):
    for column in data.columns:
        if data[column].dtype in [np.int64, np.float64]:  # Check if column is numeric
            column_median = data[column].fillna(data[column].median())
            data[column] = column_median
    return data

def train_model(data):
    # Extract the features and target variable
    features = data[['objawy', 'wiek', 'choroby_wsp', 'wzrost', 'leki']]
    target = data['zdrowie']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model

def main():

    # Read the CSV file
    data = pd.read_csv("DSP_13.csv")

    # Fill empty rows with column medians
    data_filled = fill_empty_rows_with_medians(data)

    # Train the model
    model = train_model(data_filled)

    st.set_page_config(page_title="Titanic App")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg")

    with overview:
        st.title("Titanic App")

    with right:
        age_slider = st.slider("wiek", value=1, min_value=1, max_value=90)
        sibsp_slider = st.slider("choroby wsp", min_value=0, max_value=10)
        parch_slider = st.slider("wzrost", min_value=1, max_value=300)
        fare_slider = st.slider("objawy", min_value=1, max_value=5)


    data = [[age_slider,sibsp_slider, parch_slider, fare_slider]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba przeżyłaby katastrofę?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()