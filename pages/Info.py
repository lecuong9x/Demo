import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit.components.v1 as stc
import seaborn as sns
import plotly.express as px

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def about():
    st.header("Header for about")
    df = pd.read_csv("https://raw.githubusercontent.com/Sonalikhasyap15/Diabetes_Prediction/master/diabetes_data_upload.csv")
    col1, col2 = st.columns([2,1])

    with col1:
         with st.expander("abc"): 

                fig = plt.figure()
                sns.countplot(x ='Gender', data =df, palette=random.choice(pallete))
                st.pyplot(fig)

#if __name__ == "__main__":#
#    about()
