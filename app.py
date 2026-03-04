# app.py

import streamlit as st
import pickle

# 1. Load the saved model and vectorizer
with open('model/spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 2. Page design
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 Spam Message Classifier")
st.write("Type any message below and I'll tell you if it's SPAM or NOT SPAM!")

# 3. Input box for user
user_input = st.text_area("Enter your message here:", height=150)

# 4. Predict button
if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        # 5. Convert input to numbers and predict
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        # 6. Show result
        if prediction == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT SPAM!")
