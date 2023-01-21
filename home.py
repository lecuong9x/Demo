import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit.components.v1 as stc
import seaborn as sns
import plotly.express as px
import numpy as np
#load ML
import joblib
import os


# đường link dữ liệu
# https://github.com/ASR373/diabetes-risk-prediction-app

# Contents of ~/my_app/streamlit_app.py
html_temp = """
        <div style="background-color:#3872fb; padding:10px;border-radius:10px;">
         <h1 style ="color:white; text-align:center;">Ứng dụng dữ liệu </h1>  
         <h4 style ="color:white; text-align:center;">Bệnh tiểu đường </h4> 
        </div>
            """

desc_temp = """
            ### Ứng dụng dự đoán bệnh tiểu đường
            Dữ liệu chứa các dấu hiệu và cảnh báo sớm về người mắc bệnh tiểu đường
            ### Nội dung của ứng dụng
                - Đọc dữ liệu: Phân tích dữ liệu
                - Mô hình hóa: Ứng dụng mô hình hóa dự đoán
            """

#<h4 style ="color:white, text-align:center;">Bệnh tiểu đường </h4>
def main(): # trang chính
    #st.markdown("# Main APP 🎈")
    stc.html(html_temp)
    st.markdown(desc_temp, unsafe_allow_html=True)
def contract(): # trang đọc dữ liệu
    stc.html(html_temp)
    st.markdown("# Page 2 ❄️")
    #st.sidebar.markdown("# Page 2 ❄️")
    def load_data(data):
        df = pd.read_csv(data)
        return df

    st.header("Khám phá dữ liệu")
    df = pd.read_csv("https://raw.githubusercontent.com/ASR373/diabetes-risk-prediction-app/master/diabetes_data_upload.csv")
    #df= load_data("https://raw.githubusercontent.com/ASR373/diabetes-risk-prediction-app/master/diabetes_data_upload.csv")
    #df= load_data("D:\Cuong_DA\Diabetes_dataset.xlsx")
    freq_df = pd.read_csv("https://raw.githubusercontent.com/ASR373/diabetes-risk-prediction-app/master/freqdist_of_age_data.csv")
    df_encoded = load_data("https://raw.githubusercontent.com/ASR373/diabetes-risk-prediction-app/master/diabetes_data_upload_clean.csv")
    submenu = st.sidebar.selectbox("Trang phụ", ["Diễn giải", "Plots"])
    if submenu == "Diễn giải":
        st.dataframe(df)

        with st.expander("Loại dữ liệu"):
            st.dataframe(df.dtypes)
        
        with st.expander("Diễn giải dữ liệu"):
            st.dataframe(df.describe())

        with st.expander("Phân bổ số lường trường đánh giá (Class)"):
            st.dataframe(df["class"].value_counts())

        with st.expander("Phân bổ số lường giới tính(Gender)"):
            st.dataframe(df["Gender"].value_counts())

        with st.expander("Các trường trống"):
            st.dataframe(df.isnull().sum())

    elif submenu == "Plots":
        st.subheader("Biểu đồ")

        col1, col2 = st.columns([2,1])

        with col1:
            with st.expander("Biểu đồ PLKH theo giới tính"): 
                
                gen_df  = df['Gender'].value_counts()
                gen_df = gen_df.reset_index()
                gen_df.columns = ["Gender Type", "Counts"]
                
                pl = px.pie(gen_df, names = "Gender Type",values = "Counts")
                st.plotly_chart(pl,use_container_width=True)
                
            with st.expander("Biểu đồ theo PPLKH"):
                class_count  = df['class'].value_counts()
                
        #fig = plt.figure(figsize=(10,5))
                chartdata = pd.DataFrame(class_count)
                st.bar_chart(chartdata)

                
        with col2:
            with st.expander("PLKH theo giới tính"):
                st.dataframe(gen_df)

            with st.expander("PLKH theo Hạng khách hàng"):
                st.dataframe(class_count)
        
        with st.expander("Phân phối của độ tuổi"):
            #st.write(freq_df)    
            p2= px.bar(freq_df,x="s", y="count")
            st.plotly_chart(p2)

        with st.expander("Biểu đồ ngoại vi"):
            fig = plt.figure()
            #sns.boxenplot(df["Age"])
            sns.boxplot(df["Age"])
            st.pyplot(fig)

            p3=px.box(df,x = "Age",color="Gender")
            st.plotly_chart(p3)

        # correlation
        with st.expander("Biểu đồ tương quan"):
            corr_matrix = df_encoded.corr()
            fig = plt.figure(figsize=(20,10))
            sns.heatmap(corr_matrix,annot = True)
            st.pyplot(fig)

        # Cách khác vẽ corr
        p4 = px.imshow(corr_matrix)
        st.plotly_chart(p4)


def about(): # trang mô hình hóa
    #st.markdown("# Page 3 🎉")
    #st.sidebar.markdown("# Page 3 🎉")
    attrib_info = """
    - Age 1.20-65.
    - Sex: 1. Male, 2.Female.
    - Polyuria: 1.Yes, 2.No.
    - Polydipsia: 1.Yes, 2.No.
    - Sudden weight loss: 1.Yes, 2.No.
    - Weakness: 1.Yes, 2.No.
    - Polyphagia: 1.Yes, 2.No.
    - Genital thrush: 1.Yes, 2.No.
    - Visual blurring: 1.Yes, 2.No.
    - Itching: 1.Yes, 2.No.
    - Irritability: 1.Yes, 2.No.
    - Delayed healing: 1.Yes, 2.No.
    - Partial paresis: 1.Yes, 2.No.
    - Muscle stiffness: 1.Yes, 2.No.
    - Alopecia: 1.Yes, 2.No.
    - Obesity: 1.Yes, 2.No.
    - Class: 1.Positive, 2.Negative.
    """
    label_dict = {"No":0,"Yes":1}
    gender_map =  {"Female":0,"Male":1}
    target_label_map = {"Negative":0,"Positive":1}

    def get_fvalue(val):
        feature_dict = {"No":0,"Yes":1}
        for key,value in feature_dict.items():
            if val == key:
                return value
    
    def get_value(val,my_dict):
        for key,value in my_dict.items():
            if val == key:
                return value

    #Load ML model
    @st.cache
    def load_model(model_file):
        loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
        return loaded_model
    
    def run_ml_app():
        st.subheader("ML Prediction")

        with st.expander("Thông tin thuộc tính"):
            st.markdown(attrib_info)

    #layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age",10,100)# tạo 1 range tuổi từ 10 đến 100
        gender = st.radio("Gender",("Male","Female"))
        polyuria = st.radio("Polyuria",["No","Yes"])
        polydipsia = st.radio("Polydipsia",["No","Yes"])
        sudden_weight_loss =  st.radio("Sudden_weight_loss",["No","Yes"])
        weakness = st.radio("Weakness",["No","Yes"])
        polyphagia = st.radio("polyphagia",["No","Yes"])
        genital_thrush = st.radio("Genital_thrush",["No","Yes"])
    
    with col2:
        visual_blurring = st.selectbox("Visual_blurring",["No","Yes"])
        itching = st.radio("itching",["No","Yes"]) 
        irritability = st.radio("irritability",["No","Yes"]) 
        delayed_healing =  st.radio("delayed_healing",["No","Yes"]) 
        partial_paresis = st.selectbox("partial_paresis",["No","Yes"]) 
        muscle_stiffness =  st.radio("muscle_stiffness",["No","Yes"]) 
        alopecia = st.radio("alopecia",["No","Yes"]) 
        obesity = st.select_slider("obesity",["No","Yes"]) 

    with st.expander("Your selected Options"):
        result = {"age":age,
        "gender":gender,
        "polyuria":polyuria,
        "polydipsia":polydipsia,
        "sudden_weight_loss":sudden_weight_loss,
        "weakness":weakness,
        "polyphagia":polyphagia,
        "genital_thrush":genital_thrush,
        "visual_blurring":visual_blurring,
        "itching":itching,
        "irritability":irritability,
        "delayed_healing":delayed_healing,
        "partial_paresis":partial_paresis,
        "muscle_stiffness":muscle_stiffness,
        "alopecia":alopecia,
        "obesity":obesity}

        st.write(result)

        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            elif i in ["Female","Male"]:
                res = get_value(i,gender_map)
                encoded_result.append(res)
            else:
                encoded_result.append(get_fvalue(i))
    
        st.write(encoded_result)

page_names_to_funcs = {
    "Trang Chính": main,
    "Đọc dữ liệu": contract,
    "Mô hình hóa": about,
}
selected_page = st.sidebar.selectbox("Lựa chọn trang", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




#def main():
##    st.title("Ứng dụng phân tích")
#    menu = ["Trang chủ", "Đọc dữ liệu","Mô hình dự đoán","About"]
#    choice= st.sidebar.selectbox("Lựa chọn",menu)

#    if choice == "Trang chủ":
#        st.subheader("Trang chủ")
#    elif choice == "Đọc dữ liệu":
#         run_eda_app ()
#    elif choice == "Mô hình dự đoán":
#        pass
#    else:
#        st.subheader("About") 


#if __name__ == "__main__":
#    main()