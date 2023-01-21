import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
def run_eda_app():
    menu = ["Trang chủ", "about",]
    choice= st.sidebar.selectbox("Lựa chọn",menu)
    
    if choice == "Trang chủ":
        st.subheader("Trang chủ")
    


        st.header("Khám phá dữ liệu")
        df = pd.read_csv("https://raw.githubusercontent.com/Sonalikhasyap15/Diabetes_Prediction/master/diabetes_data_upload.csv")
        st.dataframe(df)

        val_count  = df['Gender'].value_counts()
        st.write(val_count)
        #fig = plt.figure(figsize=(10,5))
        chartdata = pd.DataFrame(val_count)
        st.bar_chart(chartdata)
        #fig.title('Some title')
        #fig.ylabel('y label', fontsize=12)
        #fig.xlabel('x label', fontsize=12)


        # Add figure in streamlit app
        #st.pyplot(fig)


        # Add figure in streamlit app
        #st.pyplot(fig)
        
    #elif choice == "Đọc dữ liệu":
        # about ()
if __name__ == "__main__":
    run_eda_app()

