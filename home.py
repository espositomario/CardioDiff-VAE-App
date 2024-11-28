import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import os
from plotly.graph_objects import Figure, Violin
import plotly.graph_objects as go
from streamlit_pdf_viewer import pdf_viewer

# Streamlit app layout
st.set_page_config(layout="wide")


# Load gene cluster dictionary
with open(f'./data/gene_clusters_dict.pkl', 'rb') as f:
    GENE_CLUSTERS = pickle.load(f)

# Load DATA
DATA = pd.read_csv(f'./data/DATA.csv', index_col='GENE')

#
RNA_FPKM= pd.read_csv(f'./data/RNA_FPKMs.csv', index_col='GENE')


# List of continuous features for coloring
continuous_features = ["RNA_CV", "VAE_RMSE", "VAE_Sc"]


