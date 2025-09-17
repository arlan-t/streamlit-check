import streamlit as st

st.title("Hello Streamlit + GitHub!")
st.write("This is my first app connected with GitHub 🚀")

name = st.text_input("What's your name?")
if name:
    st.success(f"Hi {name}, welcome to my app!")
