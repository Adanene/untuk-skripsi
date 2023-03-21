import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
####ship stability pdf
#https://docs.google.com/spreadsheets/d/1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y/edit#gid=0
import pandas as pd

import streamlit as st

sheet_id ='1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y/export?format-xlsx")
df = pd.read_excel(xls , 'Sheet1' , header = 0)
df = df[["Jenis Kapal", "Loa", "Lwl", "Breadth", "Depth", "Draft", "Cb", "beban A", "beban B" , "beban C", "beban D", "beban E", "beban F" , "Jumlah beban", "kiri", "kanan", "incl depan", "Incl belakang"]]
df.head()

df['Jenis Kapal'].value_counts()
df = df.dropna()
df.isnull().sum()


df["Loa"].unique()
df["Draft"].unique()
df["Lwl"].unique()
df["Breadth"].unique()
df["Depth"].unique()
df["Cb"].unique()
df["beban A"].unique()
df["beban B"].unique()
df["beban C"].unique()
df["beban D"].unique()
df["beban E"].unique()
df["beban F"].unique()
df["Jumlah beban"].unique()
df["kiri"].unique()
df["kanan"].unique()

from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
le_loa = LabelEncoder()
df['Loa'] = le_loa.fit_transform(df['Loa'])
df["Loa"].unique()
#le.classes_

le_kapal = LabelEncoder()
df['Jenis Kapal'] = le_kapal.fit_transform(df['Jenis Kapal'])
df["Jenis Kapal"].unique()

le_draft= LabelEncoder()
df['Draft'] = le_draft.fit_transform(df['Draft'])
df["Draft"].unique()


le_lwl= LabelEncoder()
df['Lwl'] = le_lwl.fit_transform(df['Lwl'])
df["Lwl"].unique()


le_Breadth= LabelEncoder()
df['Breadth'] = le_Breadth.fit_transform(df['Breadth'])
df["Breadth"].unique()

le_Depth= LabelEncoder()
df['Depth'] = le_Depth.fit_transform(df['Depth'])
df["Depth"].unique()

le_Cb= LabelEncoder()
df['Cb'] = le_Cb.fit_transform(df['Cb'])
df["Cb"].unique()

le_bebanA= LabelEncoder()
df['beban A'] = le_bebanA.fit_transform(df['beban A'])
df["beban A"].unique()

le_bebanB= LabelEncoder()
df['beban B'] = le_bebanB.fit_transform(df['beban B'])
df["beban B"].unique()

le_bebanC= LabelEncoder()
df['beban C'] = le_bebanC.fit_transform(df['beban C'])
df["beban C"].unique()

le_bebanD= LabelEncoder()
df['beban D'] = le_bebanD.fit_transform(df['beban D'])
df["beban D"].unique()

le_bebanE= LabelEncoder()
df['beban E'] = le_bebanE.fit_transform(df['beban E'])
df["beban E"].unique()

le_bebanF= LabelEncoder()
df['beban F'] = le_bebanF.fit_transform(df['beban F'])
df["beban F"].unique()

le_jumbeb= LabelEncoder()
df['Jumlah beban'] = le_jumbeb.fit_transform(df['Jumlah beban'])
df["Jumlah beban"].unique()

le_kiri= LabelEncoder()
df['kiri'] = le_kiri.fit_transform(df['kiri'])
df["kiri"].unique()

le_kanan= LabelEncoder()
df['kanan'] = le_kanan.fit_transform(df['kanan'])
df["kanan"].unique()


X = df.drop(["incl depan", "Incl belakang"], axis=1)

y = df[['incl depan','Incl belakang']]

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)

y_pred = linear_reg.predict(X)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))

from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)

y_pred = dec_tree_reg.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))

from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)

y_pred = random_forest_reg.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))

from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)

regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))

# "Jenis Kapal", "Loa", "Lwl", "Breadth", "Depth", "Draft", "Cb", "beban A", "beban B" , "beban C", "beban D", "beban E", "beban F" , "Jumlah beban", "kiri", "kanan",
X = np.array([['Kapal penumpang', 62.8, 58.320, 12, 4, 2.7, 0.682, 4.003, 3.718, 3.95, 4.022, 0, 0, 4,'AB','CD']])
X[:, 0] = le_kapal.transform(X[:,0])
X[:, 14] = le_kiri.transform(X[:,14])
X[:, 15] = le_kanan.transform(X[:,15])
X = X.astype(float)

y_pred = regressor.predict(X)
y_pred

import pickle
data = {"model": regressor, "le_kapal": le_kapal, "le_loa": le_loa, "le_lwl": le_lwl, "le_Breadth": le_Breadth, "le_Depth": le_Depth, "le_draft": le_draft, "le_Cb": le_Cb, "le_bebanA": le_bebanA, "le_bebanB": le_bebanB, "le_bebanC": le_bebanC, "le_bebanD": le_bebanD, "le_bebanE": le_bebanE, "le_bebanF": le_bebanF, "le_jumbeb": le_jumbeb, "le_kiri": le_kiri, "le_kanan": le_kanan}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_kapal = data["le_kapal"]
le_loa = data["le_loa"]
le_lwl = data["le_lwl"]
le_Breadth = data["le_Breadth"]
le_Depth = data["le_Depth"]
le_draft = data["le_draft"]
le_Cb = data["le_Cb"]
le_bebanA = data["le_bebanA"]
le_bebanB = data["le_bebanB"]
le_bebanC = data["le_loa"]
le_bebanD = data["le_bebanD"]
le_bebanE = data["le_bebanE"]
le_bebanF = data["le_bebanF"]
le_jumbeb = data["le_jumbeb"]
le_kiri = data["le_kiri"]
le_kanan = data["le_kanan"]
y_pred = regressor_loaded.predict(X)
y_pred


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
        
  
page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
  
