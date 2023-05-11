import streamlit as st
import pickle
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

startTime = datetime.now()

# import znanych nam bibliotek

data = pd.read_csv("DSP_13.csv", delimiter=';')
# Extract the features and target variable
features = data[['objawy', 'wiek', 'choroby_wsp', 'wzrost', 'leki']]
target = data['zdrowie']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Create an imputer object with strategy='median'
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training data and transform it
X_train = imputer.fit_transform(X_train)

# Transform the testing data using the fitted imputer
X_test = imputer.transform(X_test)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

def main():

	st.set_page_config(page_title="Titanic App")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg")

	with overview:
		st.title("Titanic App")

	with right:
		objawy_slider = st.slider("Objawy", value=1, min_value=0, max_value=10)
		wiek_slider = st.slider("Wiek", value=1, min_value=0, max_value=120)
		choroby_wsp_slider = st.slider("Choroby współistniejące", value=1, min_value=0, max_value=5)
		wzrost_slider = st.slider("Wzrost", value=1, min_value=120, max_value=220)
		leki_slider = st.slider("Leki", value=1, min_value=0, max_value=10)

	# read the data
	df = pd.read_csv('DSP_13.CSV')
	# remove the missing values
	df.dropna(inplace=True)
	# take the median of each column
	medians = df.median()
	# replace the missing values with the medians
	objawy_slider = medians['objawy'] if objawy_slider is None else objawy_slider
	wiek_slider = medians['wiek'] if wiek_slider is None else wiek_slider
	choroby_wsp_slider = medians['choroby_wsp'] if choroby_wsp_slider is None else choroby_wsp_slider
	wzrost_slider = medians['wzrost'] if wzrost_slider is None else wzrost_slider
	leki_slider = medians['leki'] if leki_slider is None else leki_slider

	data = [[objawy_slider, wiek_slider, choroby_wsp_slider, wzrost_slider, leki_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
