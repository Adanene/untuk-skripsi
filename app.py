import streamlit as st
predict_page = st.file_uploader(predict_page.py)
explore_page = st.file_uploader(explore_page.py)
    
    
from predict_page import show_predict_page
from explore_page import show_explore_page
    

    
page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
