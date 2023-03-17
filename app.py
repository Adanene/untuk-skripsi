import streamlit as st
import pandas as pd
import numpy as np
 
from predict_page import show_predict_page
from explore_page import show_explore_page
    

    
page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
  
#for explore page
def explore_page():
 import streamlit as st
 import pandas as pd
 import matplotlib.pyplot as plt

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


@st.cache
def load_data():
    df = pd.read_excel(f"https://docs.google.com/spreadsheets/d/1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y/export?format-xlsx" , 'Sheet1' , header = 0)
    return df
  
def show_explore_page():
    st.title("Explore Ship Stability")

    st.write(
        """
    ### Inclining test adalah percobaan kemiringan yang harus dilakukan untuk mengetahui berat dan letak titik berat kapal kosong setelah selesai dibangun (Biro Klasifikasi Indonesia,2003). Pengujian ini dilakukan untuk memenuhi persyaratan class dalam rangkan pemenuhan persyaratan statutory untuk Badan Pemerintah.
    """
    )

def predict_page(): 
  import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_kapal = data["le_kapal"]
le_loa = data["le_loa"]
le_lwl = data["le_lwl"]
le_draft = data["le_draft"]
le_Breadth = data["le_Breadth"]
le_Depth = data["le_Depth"]
le_Cb = data["le_Cb"]
le_bebanA = data["le_bebanA"]
le_bebanB = data["le_bebanB"]
le_bebanC = data["le_bebanC"]
le_bebanD = data["le_bebanD"]
le_bebanE = data["le_bebanE"]
le_bebanF = data["le_bebanF"]
le_jumbeb = data["le_jumbeb"]
le_kiri = data["le_kiri"]
le_kanan = data["le_kanan"]

#for show_predict_page
def show_predict_page():
    st.title("Ship Inclining Test Prediction")

    st.write("""### We need some information to predict the incline""")

    Kapal = (
        "Kapal penumpang",
        "Kapal patroli",
        "Kapal kargo",
    )

    jumlah_beban = (
        "4",
        "6",
    )

    kiri4 = (
        "AB",
        "B",
        "None",
        "C",
        "CD",
        "ACD",
        "ABCD",
        "ABD",   
    )

    kanan4 = (
        "CD",
        "ACD",
        "ABCD",
        "ABD",
        "AB",
        "B",
        "None",
        "C",  
    )

    kiri6 = (
        "ABC",
        "BC",
        "C",
        "None",
        "ABC"
        "ABCD",
        "ABCDE",
        "ABCDEF", 
    )

    kanan6 = (
        "DEF",
        "ADEF",
        "ABDEF",
        "ABCDEF",
        "DEF",
        "EF",
        "F",
        "None",
    )

    Kapal = st.selectbox("Jenis Kapal", Kapal)
    Loa = st.number_input("Length Over All (m)", step =0.01)
    Lwl = st.number_input("Length Water Line (m)", max_value= Loa)
    Breadth = st.number_input("Breadth (m)", step =0.01)
    Depth = st.number_input("Depth (m) ", step =0.01)
    Draft = st.number_input("Draft (m) ", max_value= Depth, step =0.01)
    Cb = st.number_input("Coefficient Block", min_value= 0.00, max_value= 1.00, step =0.01)
    beban_A = st.number_input("Beban A (Ton)",min_value= 0.00,  step =0.01)
    beban_B = st.number_input("Beban B (Ton)",min_value= 0.00,  step =0.01)
    beban_C = st.number_input("Beban C (Ton)",min_value= 0.00,  step =0.01)
    beban_D = st.number_input("Beban D (Ton)",min_value= 0.00,  step =0.01)
    jumlah_beban = st.selectbox("Jumlah beban uji", jumlah_beban)

    if jumlah_beban == "4" :
        beban_E = 0
        beban_F = 0
        proses = st.slider("Proses incline", 0, 7, 0)
        if proses == 0:
            kiri = "AB"
            kanan = "CD"
        if proses == 1:
            kiri = "B"
            kanan = "ACD"
        if proses == 2:
            kiri = "None"
            kanan = "ABCD"
        if proses == 3:
            kiri = "C"
            kanan = "ABD"
        if proses == 4:
            kiri = "CD"
            kanan = "AB"
        if proses == 5:
            kiri = "ABC"
            kanan = "D"
        if proses == 6:
            kiri = "ABCD"
            kanan = "None"
        if proses == 7:
            kiri = "C"
            kanan = "ABD"
        st.write("beban kiri :", kiri)
        st.write("beban kanan :", kanan)

    
    if jumlah_beban == "6" :
        beban_E = st.number_input("Beban E (Ton)",min_value= 0.00, step =0.01)
        beban_F = st.number_input("Beban F (Ton)",min_value= 0.00, step =0.01)
        proses = st.slider("Proses incline", 0, 7, 0)
        if proses == 0:
            kiri = "ABC"
            kanan = "DEF"
        if proses == 1:
            kiri = "BC"
            kanan = "ADEF"
        if proses == 2:
            kiri = "C"
            kanan = "ABDEF"
        if proses == 3:
            kiri = "None"
            kanan = "ABCDEF"
        if proses == 4:
            kiri = "ABC"
            kanan = "DEF"
        if proses == 5:
            kiri = "ABCD"
            kanan = "EF"
        if proses == 6:
            kiri = "ABCDE"
            kanan = "F"
        if proses == 7:
            kiri = "ABCDEF"
            kanan = "None"
        st.write("beban kiri :", kiri)
        st.write("beban kanan :", kanan)

    ok = st.button("Calculate Incline")
    if ok:
        X = np.array([[Kapal, Loa, Lwl, Breadth, Depth, Draft, Cb, beban_A, beban_B, beban_C, beban_D, beban_E, beban_F,  jumlah_beban, kiri, kanan, ]])
        X[:, 0] = le_kapal.transform(X[:,0])
        X[:, 14] = le_kiri.transform(X[:,14])
        X[:, 15] = le_kanan.transform(X[:,15])
        X = X.astype(float)

        salary = regressor.predict(X)

        st.subheader(f"Ship will turn in {salary} degrees")
