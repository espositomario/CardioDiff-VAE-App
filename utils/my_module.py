import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
from plotly.graph_objects import Figure, Violin
import plotly.graph_objects as go
from streamlit_pdf_viewer import pdf_viewer

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)



CT_LIST = ['ESC', 'MES', 'CP', 'CM']
HM_LIST = ['H3K4me3', 'H3K27ac', 'H3K27me3',  'RNA']
PREFIXES = [HM + '_' + CT for HM in HM_LIST for CT in CT_LIST]


MARKER_GENES_EXT = {'ESC': ['Nanog','Pou5f1','Sox2','L1td1','Dppa5a','Tdh','Esrrb','Lefty1','Zfp42','Sfn','Lncenc1','Utf1'],
                    'MES': ['Mesp1','Mesp2','T', 'Vrtn','Dll3','Dll1', 'Evx1','Cxcr4','Pcdh8','Pcdh19','Robo3','Slit1'],
                    'CP':  ['Sfrp5', 'Gata5', 'Tek','Hbb-bh1','Hba-x', 'Pyy','Sox18','Lyl1','Rgs4','Igsf11','Tlx1','Ctse'],
                    'CM':  ['Nppa','Gipr', 'Actn2', 'Coro6', 'Col3a1', 'Bgn','Myh6','Myh7','Tnni3','Hspb7' ,'Igfbp7','Ndrg2'],
                    }



HM_COL_DICT = {'H3K4me3': '#f37654','H3K27ac': '#b62a77','H3K27me3': '#39A8AC','RNA':'#ED455C'}
CT_COL_DICT= {'ESC': '#405074',
                'MES': '#7d5185',
                'CP': '#c36171',
                'CM': '#eea98d',}

CV_COL_DICT= {'RNA_ESC': '#405074',
                'RNA_MES': '#7d5185',
                'RNA_CP': '#c36171',
                'RNA_CM': '#eea98d',
                'STABLE':'#B4CD70',
                'other':'#ECECEC'}
GONZALEZ_COL_DICT= {'Active': '#E5AA44','Bivalent': '#7442BE','other':'#ECECEC'}

COLOR_DICTS = {
    "CV_Category": CV_COL_DICT,
    "ESC_ChromState_Gonzalez2021": GONZALEZ_COL_DICT,
}



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    From:(https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/)
    
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters")

    #if not modify:
    #    return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Select columns by which filter genes", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def df_tabs(DF):
    # Feature groups
    Z_AVG = DF[Z_AVG_features]
    LOG_FC = DF[LOG_FC_features]
    MISC = DF[MISC_features]
    LATENT = DF[LATENT_features]
    FPKM = DF[FPKM_features]

    # Create Streamlit tabs
    tabs = st.tabs(["Whole Table", "Features Z-scores", "RNA Log FCs", "Annotations", "VAE Latent Space", "RNA FPKMs"])

    with tabs[0]:  # All
        st.markdown("### Whole Table", 
                    help="This table contains all the input features, annotations, and other information for all genes.")
        st.dataframe(filter_dataframe(DF))

    with tabs[1]:  # Z-scores
        st.markdown("### Input features Z-scores", 
                    help="This table displays the Z-scores of ChIP-seq levels, which represent chromatin immunoprecipitation sequencing data averaged between replicates.")
        st.dataframe(filter_dataframe(Z_AVG))

    with tabs[2]:  # Log Fold Changes
        st.markdown("### Input RNA Log Fold Changes (LogFCs)", 
                    help="This table contains the log fold change values, showing the relative expression differences between experimental conditions.")
        st.dataframe(filter_dataframe(LOG_FC))

    with tabs[3]:  # Annotations
        st.markdown("### Gene annotations", 
                    help="This table includes annotations and miscellaneous features for the genes, such as genomic context, functional categories, or metadata.")
        st.dataframe(filter_dataframe(MISC))

    with tabs[4]:  # VAE Latent Space
        st.markdown("### VAE Latent Variables and UMAP projections", 
                    help="These columns represent the 6D latent space coordinates derived from a Variational Autoencoder (VAE). They capture compressed representations of gene features.")
        st.dataframe(filter_dataframe(LATENT))

    with tabs[5]:  # RNA FPKMs
        st.markdown("### RNA-seq FPKMs per replicate", 
                    help="This table contains RNA-seq FPKM (Fragments Per Kilobase of transcript per Million mapped reads) values for gene expression across different cell types.")
        st.dataframe(filter_dataframe(FPKM))

def scatter(DATA, selected_feature,key, COLOR_DICTS=None):
    CAT = False
    if pd.api.types.is_categorical_dtype(DATA[selected_feature]) or DATA[selected_feature].dtype == 'object':
        CAT = True
        color_dict = COLOR_DICTS.get(selected_feature, None)
        
    with st.popover("⚙️", ):
        point_size = st.slider("Point Size", min_value=1, max_value=8, value=3, step=1, key=key+'point_size')
        point_opacity = st.slider("Transparency", min_value=0.1, max_value=1.0, value=0.8, step=0.1, key=key+'point_opacity')
        if not CAT:
            colormap = st.segmented_control("Select Colormap", ["Turbo", "Blues","viridis", "RdBu_r"], default= 'Turbo',selection_mode='single', key=key+'colormap')
            min_col, max_col = st.slider("Color Range (percentile)", min_value=1, max_value=99, value=(1,99), step=1, key=key+'min_max')

    # Determine if the selected feature is categorical or continuous
    if CAT:
        # Handle categorical feature
        fig = px.scatter(
            DATA,
            x="VAE_UMAP1",
            y="VAE_UMAP2",
            color=selected_feature,
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
            color_discrete_map=color_dict if color_dict else None
        )
    else:
        # Handle continuous feature
        # Compute 1st and 99th percentiles for color scaling
        

        min_p, max_p = np.percentile(DATA[selected_feature], [min_col, max_col])
        fig = px.scatter(
            DATA,
            x="VAE_UMAP1",
            y="VAE_UMAP2",
            color=selected_feature,
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
            color_continuous_scale=colormap,
            range_color=(min_p, max_p),  # Apply percentile scaling
        )
    
    
    
    fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))

    fig.update_layout(
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_tickvals=[], yaxis_tickvals=[],
        plot_bgcolor="white", autosize=True
    )
    return fig



# Load gene cluster dictionary
with open(f'./data/gene_clusters_dict.pkl', 'rb') as f:
    GENE_CLUSTERS = pickle.load(f)

# Load DATA
DATA = pd.read_csv(f'./data/DATA.csv', index_col='GENE')
DATA['Cluster'] = pd.Categorical(DATA['Cluster'])

#
RNA_FPKM= pd.read_csv(f'./data/RNA_FPKMs.csv', index_col='GENE')


# List of continuous features for coloring
continuous_features = ["RNA_CV", "VAE_RMSE", "VAE_Sc"]


Z_AVG_features = ['RNA_ESC', 'RNA_MES', 'RNA_CP', 'RNA_CM', 'H3K4me3_ESC', 'H3K4me3_MES',
        'H3K4me3_CP', 'H3K4me3_CM', 'H3K27ac_ESC', 'H3K27ac_MES', 'H3K27ac_CP',
        'H3K27ac_CM', 'H3K27me3_ESC', 'H3K27me3_MES', 'H3K27me3_CP',
        'H3K27me3_CM']
LOG_FC_features = ['RNA_CM_CP_FC', 'RNA_CM_MES_FC', 'RNA_CM_ESC_FC',
            'RNA_CP_MES_FC', 'RNA_CP_ESC_FC', 'RNA_MES_ESC_FC']

MISC_features = [ 'VAE_RMSE', 'VAE_Sc', 'RNA_CV', 'CV_Category', 'ESC_ChromState_Gonzalez2021', 'Cluster']

LATENT_features = ['VAE1', 'VAE2', 'VAE3', 'VAE4', 'VAE5', 'VAE6', 'VAE_UMAP1', 'VAE_UMAP2']

FPKM_features = [ 'RNA_ESC_1', 'RNA_ESC_2', 'RNA_MES_1', 'RNA_MES_2',
            'RNA_CP_1', 'RNA_CP_2', 'RNA_CM_1', 'RNA_CM_2']


COLOR_FEATURES = MISC_features + Z_AVG_features + LOG_FC_features

