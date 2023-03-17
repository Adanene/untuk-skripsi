#https://docs.google.com/spreadsheets/d/1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y/edit#gid=0
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sheet_id ='1vSeqmo559uXl_sjwCorKsENK2boOihERFrmrM-V5D2Y'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format-xlsx")
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




