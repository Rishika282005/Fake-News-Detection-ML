import streamlit as st
import pickle

model = pickle.load(open("models/model.pkl","rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl","rb"))

st.title("Fake News Detection System")

news = st.text_area("Enter news article")

if st.button("Check News"):

    vect = vectorizer.transform([news])

    prediction = model.predict(vect)

    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")