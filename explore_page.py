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

df = load_data()

def show_explore_page():
    st.title("Explore Ship Stability")

    st.write(
        """
    ### Inclining test adalah percobaan kemiringan yang harus dilakukan untuk mengetahui berat dan letak titik berat kapal kosong setelah selesai dibangun (Biro Klasifikasi Indonesia,2003). Pengujian ini dilakukan untuk memenuhi persyaratan class dalam rangkan pemenuhan persyaratan statutory untuk Badan Pemerintah.
    """
    )
