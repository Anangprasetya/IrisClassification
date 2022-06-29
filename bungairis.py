import streamlit as st
import pandas as pd
import joblib
from PIL import Image

model= open("DecisionTreeClassifier.pkl", "rb")
knn_clf=joblib.load(model)

st.title("Iris flower species Classification App")
#Loading images
setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')

st.sidebar.title("Features")
#Intializing
parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]



#Display
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
    values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
    parameter_input_values.append(values)
 
 
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')

if st.button("Click Here to Classify"):
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




#referensi   : https://towardsdatascience.com/beginners-guide-lets-make-an-interactive-iris-flower-classification-app-using-streamlit-42e1026d2167
#            : https://www.educative.io/answers/how-to-build-a-decision-tree-with-the-iris-dataset-in-python