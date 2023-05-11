import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def fill_empty_rows_with_medians(data):
    for column in data.columns:
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
	data = pd.read_csv("DSC_13.csv")

	# Fill empty rows with column medians
	data_filled = fill_empty_rows_with_medians(data)

	# Save the filled data back to the CSV file
	data_filled.to_csv("DSC_13_filled.csv", index=False)

	# Train the model
	model = train_model(data_filled)

	st.set_page_config(page_title="Titanic App")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg")

	with overview:
		st.title("Titanic App")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x])
		pclass_radio = st.radio("Class",list(pclass_d.keys()),index = 2,format_func = lambda x:pclass_d[x])
		embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x])

	with right:
		age_slider = st.slider("Wiek", value=1, min_value=1, max_value=90)
		sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", min_value=0, max_value=10)
		parch_slider = st.slider("Liczba rodziców i/lub dzieci", min_value=0, max_value=10)
		fare_slider = st.slider("Cena biletu", min_value=0, max_value=480, step=1)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
