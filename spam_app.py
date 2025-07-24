import streamlit as st
import joblib

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📧 Spam Message Classifier")
st.write("Type a message below and find out if it's spam or not.")

message = st.text_area("Your message here:")

if st.button("Classify"):
    if message:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)[0]
        st.write("### Result:", "🚫 Spam" if prediction else "✅ Not Spam")
    else:
        st.warning("Please enter a message.")
