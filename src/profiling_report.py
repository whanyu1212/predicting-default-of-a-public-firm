import pandas as pd
import streamlit as st
from streamlit_ydata_profiling import st_profile_report
from ydata_profiling import ProfileReport

df = pd.read_csv("./data/raw/input.csv")
profile = ProfileReport(df, title="Profiling Report")
st.header(":clipboard: Profiling Report of the dataset")
st_profile_report(profile, navbar=True)
