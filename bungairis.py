import streamlit as st
import pandas as pd
import joblib
from PIL import Image

model= open("DecisionTreeClassifier.pkl", "rb")
knn_clf=joblib.load(model)

st.title("Machine Learning Klasifikasi Bunga Iris")
#Loading images
setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')

st.sidebar.title("Masukkan Data")
#Intializing
parameter_list=['Panjang Sepal (cm)','Lebar Sepal (cm)','Panjang Petal (cm)','Lebar Petal (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]



#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input_values.append(values)
 
 
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')

if st.button("Klasifikasi"):
    prediction = knn_clf.predict(input_variables)
    if prediction == 0:
        # print(f'Ini adalah sentosa brooo ============================ 0')
        st.image(setosa)
    elif prediction == 1:
        # print(f'Ini adalah versicolor brooo ============================ 1')
        st.image(versicolor)
    else:
        # print(f'ini adalah virginica')
        st.image(virginica)

else:
    st.subheader('Nama Kelompok')
    st.text("2000018047 - Anang Nur Prasetya")
    st.text("1900018300 - Gilar Syaikhu Alam")
    st.text("1900018319 - Alif Akbar")


#referensi   : https://towardsdatascience.com/beginners-guide-lets-make-an-interactive-iris-flower-classification-app-using-streamlit-42e1026d2167
#            : https://www.educative.io/answers/how-to-build-a-decision-tree-with-the-iris-dataset-in-python