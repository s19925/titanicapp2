import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pd.fillna

#uczenie modelu

data = pd.read_csv("DSP_13.csv", sep=";")
y = data.iloc[:,-1]
data.drop(['objawy','wiek','choroby_wsp', 'wzrost', 'leki', 'zdrowie'], axis = 1, inplace = True)
data(['objawy','wiek','choroby_wsp', 'wzrost', 'leki', 'zdrowie']).fillna(0)
x = data.iloc[:,0:5]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
model = RandomForestClassifier(n_estimators = 10, max_depth = 4, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# otwieramy wcześniej wytrenowany model

sex_d ={0:"Mężczyzna",1:"Kobieta"}
pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

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
