import streamlit as st
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model dari file HDF5
model = load_model('lstm_sentiment_model.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100  # Sesuaikan dengan panjang maksimum yang digunakan saat melatih model

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    prediction = model.predict(padded_sequences)[0]
    
    if prediction[0] >= 0.5:
        sentiment = 'Positive'
    elif prediction[1] >= 0.5:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, prediction

# Streamlit app setup
st.set_page_config(page_title="Sentiment Analysis", page_icon=":speech_balloon:")

st.title("Sentiment Analysis of 2024 General Election Tweets")
st.write(f"**_Model's Accuracy_** :  :green[**92.1**]% (:red[_Do not copy outright_])")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1:
    st.sidebar.header("**User Input** Sidebar")

    # Input tweet dari pengguna
    tweet_input = st.sidebar.text_area(label=":violet[**Enter a tweet for sentiment analysis**]")

    st.sidebar.write("")

    data = {
        'Tweet': tweet_input,
    }

    preview_df = pd.DataFrame(data, index=['input'])

    st.header("User Input as DataFrame")
    st.write("")
    st.dataframe(preview_df, width=1000)

    result = ":violet[-]"

    predict_btn = st.button("**Predict**", type="primary")

    st.write("")
    if predict_btn:
        sentiment, prediction = predict_sentiment(tweet_input)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        result = f":green[**{sentiment}**] (Score: {prediction:.2f})" if sentiment == 'Positive' else f":red[**{sentiment}**] (Score: {prediction:.2f})" if sentiment == 'Negative' else f":blue[**{sentiment}**] (Score: {prediction:.2f})"

    st.write("")
    st.write("")
    st.subheader("Prediction:")
    st.subheader(result)

with tab2:
    st.header("Predict multiple data:")

    sample_csv = pd.DataFrame({'Tweet': ['Sample tweet 1', 'Sample tweet 2']}).to_csv(index=False).encode('utf-8')

    st.write("")
    st.download_button("Download CSV Example", data=sample_csv, file_name='sample_tweets.csv', mime='text/csv')

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        tweets = uploaded_df['Tweet'].tolist()
        prediction_arr = [predict_sentiment(tweet)[0] for tweet in tweets]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            result = "Positive" if prediction == 'Positive' else "Negative" if prediction == 'Negative' else "Neutral"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({'Prediction Result': result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
