import streamlit as st
 
from predict_page.py import show_predict_page
from explore_page.py import show_explore_page
    

    
page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
