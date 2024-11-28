import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import os
from plotly.graph_objects import Figure, Violin
import plotly.graph_objects as go
from streamlit_pdf_viewer import pdf_viewer

# Load gene cluster dictionary
with open(f'./gene_clusters_dict.pkl', 'rb') as f:
    GENE_CLUSTERS = pickle.load(f)

# Load CODE and LOG matrices
CODE = pd.read_csv(f'./CODE.csv', index_col='GENE')

# Map cluster IDs to CODE and LOG
gene_to_cluster = {}
for cluster_id, gene_list in GENE_CLUSTERS.items():
    for gene in gene_list['gene_list']:
        gene_to_cluster[gene] = cluster_id

CODE["GMM_VAE_80"] = CODE.index.map(gene_to_cluster).astype(int)

# List of continuous features for coloring
continuous_features = ["RNA_CV", "VAE_RMSE", "VAE_Sc"]

# Streamlit app layout
st.set_page_config(layout="wide")
