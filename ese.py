from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import collections
from scipy.cluster import hierarchy
from sklearn import datasets
from random import randint
import random
import plotly.express as px
import altair as alt
import seaborn as sns

def app(data):
     dss





def load_file():
    df = pd.read_csv(file)
    st.header("Dataset Table")
    st.dataframe(df, width=1000, height=500)
    return df

     
selected = option_menu(
    menu_title = None,
    options=["Task","Solution","About"],
    icons=["list-task","caret-right-square-fill","person-circle"],
    default_index=0,
    orientation="horizontal",
)



if selected=="Task":
   st.subheader("Problem Statement : ")



elif selected == "Solution":      
    st.title("Data Analysis Tool")
    file = st.file_uploader("Enter Dataset first to Proceed", type=['csv'], accept_multiple_files=False,disabled=False)
    if file:
      data = load_file()  
      app(data)

    
                                                                
                                    

         
 

elif selected == "About":
    
    st.subheader("Guided By :")
    st.subheader("Dr. B. F. Momin Sir")
    
    


    st.write(" ")
    
    st.write(" ")
    
    st.write(" ")

    st.subheader("Student Information :")
    st.write(" ")
    
    st.write(" ")

    col1,col2 = st.columns(2)

    with col1:
        st.write("PRN NO")
        st.write("2019BTECS00074")

    with col2:
        st.write("NAME")
        st.write("Sushant Patil")

    
