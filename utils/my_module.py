import streamlit as st
import random
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
import re
from plotly.graph_objects import Figure, Violin
import plotly.graph_objects as go
#custom components
from streamlit_pdf_viewer import pdf_viewer
from streamlit_extras.bottom_container import bottom
from plotly.subplots import make_subplots
import math
from matplotlib import cm, colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
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
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Select one or multiple column conditions by which filter data", df.columns)
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

MISC_features = [ 'Cluster','RNA_CV', 'CV_Category', 'ESC_ChromState_Gonzalez2021', 'VAE_RMSE', 'VAE_Sc']

LATENT_features = ['VAE1', 'VAE2', 'VAE3', 'VAE4', 'VAE5', 'VAE6', 'VAE_UMAP1', 'VAE_UMAP2']

FPKM_features = [ 'RNA_ESC_1', 'RNA_ESC_2', 'RNA_MES_1', 'RNA_MES_2',
            'RNA_CP_1', 'RNA_CP_2', 'RNA_CM_1', 'RNA_CM_2']


COLOR_FEATURES = MISC_features + Z_AVG_features + LOG_FC_features





def plot_sankey(DATA, SEL_GENES, font_size=12, font_color="black", link_opacity=0.5):
    """
    Create a Sankey diagram with genes as the first layer and clusters as the second.
    Links between genes and clusters are colored based on clusters.

    Parameters:
    - DATA (pd.DataFrame): Input data with genes as index.
    - SEL_GENES (list): List of selected gene names.
    - font_size (int): Font size for node labels.
    - font_color (str): Font color for node labels.
    - font_family (str): Font family for node labels.
    - link_opacity (float): Opacity for the links (0.0 to 1.0).

    Returns:
    - fig: A Plotly Sankey figure.
    """
    # Filter for selected genes
    data_filtered = DATA.loc[SEL_GENES]

    # Create node labels
    gene_nodes = SEL_GENES  # Gene names as the first level
    cluster_nodes = data_filtered["Cluster"].unique().tolist()  # Clusters as the second level
    
    all_nodes = gene_nodes + cluster_nodes  # Combine all nodes
    node_map = {node: i for i, node in enumerate(all_nodes)}  # Map node name to index

    # Generate a colormap for clusters
    num_clusters = len(cluster_nodes)
    cmap = cm.get_cmap("rainbow", num_clusters)  # Use rainbow colormap
    cluster_colors = {cluster: mcolors.rgb2hex(cmap(i)) for i, cluster in enumerate(cluster_nodes)}

    # Create links
    links = []
    link_colors = []

    # Links from genes to clusters
    for gene, row in data_filtered.iterrows():
        cluster = row["Cluster"]
        rgba_color = mcolors.to_rgba(cluster_colors[cluster], alpha=link_opacity)  # Convert hex to RGBA
        rgba_str = f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})"
        links.append({
            "source": node_map[gene],
            "target": node_map[cluster],
            "value": 1  # Equal weight for all links
        })
        link_colors.append(rgba_str)

    # Define node colors
    node_colors = []
    for node in all_nodes:
        if node in gene_nodes:  # Gene nodes
            node_colors.append("silver")  # Default gray for genes
        else:  # Cluster nodes
            node_colors.append(cluster_colors[node])  # Color by cluster

    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        textfont=dict(size=font_size, color=font_color),
        #arrangement = "freeform",
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            x=nodes_xy(len(gene_nodes), len(cluster_nodes))[0],
            y=nodes_xy(len(gene_nodes), len(cluster_nodes))[1],
            color=node_colors,
            hovertemplate='%{label}<extra></extra>',
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            color=link_colors,  # Colored links based on clusters
            hovertemplate='Gene: %{source.label}<br>Cluster: %{target.label}<extra></extra>',
        )
    ))

    # Update layout
    fig.update_layout(
        title_text="",
        margin=dict(t=50, l=50, r=25, b=25),
        height=  min(70+ (len(gene_nodes) * 20), 1000) # Set a fixed height

    )
    
    return fig



def nodes_xy(len_gene_nodes, len_cluster_nodes):
    """
    Generate x and y positions for Sankey nodes.

    Parameters:
    - len_gene_nodes (int): Number of gene nodes.
    - len_cluster_nodes (int): Number of cluster nodes.

    Returns:
    - x (list): x positions for all nodes.
    - y (list): y positions for all nodes.
    """
    if len_gene_nodes < 2 or len_cluster_nodes < 2:
        return None, None

    # x positions: fixed for gene nodes and varying for cluster nodes
    x_gene = [None] * len_gene_nodes
    x_cluster = list(np.linspace(0.3, 0.8, len_cluster_nodes))

    # y positions: linearly spaced from 0 to 1
    y_gene = [None] * len_gene_nodes
    y_cluster = list(np.linspace(0.1, 0.95, len_cluster_nodes))

    # Combine x and y positions
    x = x_gene + x_cluster
    y = y_gene + y_cluster

    return x, y



def parse_gene_list(uploaded_files):
    """
    Parse uploaded files to extract a list of genes.

    Parameters:
    - uploaded_files: List of uploaded file objects.

    Returns:
    - list: A deduplicated list of gene names.
    """
    gene_list = []
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        # Split content using common delimiters
        genes = re.split(r'[,\t\n\s]+', content)
        gene_list.extend(gene.strip() for gene in genes if gene.strip())  # Remove empty or whitespace-only entries
    return list(set(gene_list))  # Deduplicate the list


def select_genes():
    SEL_GENES = []
    # Segmented control to choose between manual selection or file upload
    selection_mode = st.segmented_control(
        "Select or Upload a gene list",
        options=["Select manually", "Upload files"],
    )

    if selection_mode == "Upload files":
        # Upload gene list files
        uploaded_files = st.file_uploader(
            "Upload one or multilpe files containing gene symbols", 
            help="- Files must be UTF-8 encoded text files with gene symbols separeted by commas, tabs, spaces or newlines. \n - Gene symbols must be in RefSeq mouse format (e.g., 'Hand1'). \n - Multiple files will be merged into a single gene list. \n - Gene symbols not found in the dataset will be ignored.",
            accept_multiple_files=True
        )

        # Parse 
        if uploaded_files:
            try:
                file_genes = parse_gene_list(uploaded_files)
                
                SEL_GENES = [gene for gene in file_genes if gene in DATA.index]
            except Exception as e:
                st.error(f"Error reading files: {e}")

    elif selection_mode == "Select manually":
        # Multi-select widget for gene selection
        SEL_GENES = st.multiselect(
            "Selected genes",
            options=DATA.index,
            placeholder="Type or click to select genes",
            key="select_genes"
        )

    return SEL_GENES


def download_table(DATA):
    """
    Creates a button to download the provided DataFrame as a CSV file.

    Parameters:
    - DATA: pandas DataFrame to be downloaded.

    Returns:
    - None
    """
    # Convert the DataFrame to a CSV string
    csv_data = DATA.to_csv(index=True, index_label="GeneSymbol")

    # Create the download button
    st.download_button(
        label="All Data",
        icon=":material/download:",  
        data=csv_data,
        file_name="CardioDiffVAE_data.csv",
        mime="text/csv",  # Correct MIME type for CSV files
        key='download_table'
    )
    



def plot_violin_box(df, feature_group, ct_list, hm_col_dict, ct_col_dict, k=None, y_lab='', VMIN=None, VMAX=None):
    """
    Plots a violin and box plot for a single feature group.

    Args:
        df (pd.DataFrame): The data frame containing the data.
        feature_group (str): The feature group to be plotted (e.g., 'H3K4me3').
        ct_list (list of str): List of cell types for grouping (e.g., ['ESC', 'MES', 'CP', 'CM']).
        hm_col_dict (dict): Dictionary mapping feature groups to colors.
        ct_col_dict (dict): Dictionary mapping cell types to colors.
        k (int, optional): Cluster value to filter the DataFrame by the 'Cluster' column.
        y_lab (str): Label for the x-axis.
    """



    
    # Melt the DataFrame to long format for plotting
    melted_df = df.melt(var_name='CT', value_name='Value')
    melted_df['CT'] = melted_df['CT'].str.replace(f"{feature_group}_", '')

    # Create the plot
    fig, ax = plt.subplots(figsize=(3, 4))

    # Violin plot (not displayed as per requirements)
    # sns.violinplot(data=melted_df, x='CT', y='Value', palette=[ct_col_dict[ct] for ct in ct_list],
    #                ax=ax, inner=None, linewidth=0, saturation=1, cut=0, alpha=0.9)
    
    # Box plot overlay
    sns.boxplot(
        data=melted_df, x='CT', y='Value', palette=[ct_col_dict[ct] for ct in ct_list], 
        ax=ax, width=0.6, saturation=0.8,
        showcaps=False,
        showfliers=False,
        #boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.5),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', alpha=0.7),
        #flierprops=dict(marker='o', markerfacecolor='grey', markersize=4, alpha=0.8)
    )
    # Scatter points overlay (jittered)
    sns.stripplot(
        data=melted_df, x='CT', y='Value', palette=[ct_col_dict[ct] for ct in ct_list], 
        ax=ax, jitter=True, size=1, alpha=0.5, linewidth=0.5, 
    )




    # Customize plot
    ax.set_title(f'{feature_group}', fontsize=14, color='white', backgroundcolor=hm_col_dict.get(feature_group, 'black'),
                        pad=20)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)  # Horizontal line at y=0
    ax.set_ylim(VMIN, VMAX)  # Use global min and max for this feature group
    ax.set_ylabel(y_lab, fontsize=12)
    ax.set_xlabel('', )
    sns.despine(ax=ax)

    plt.tight_layout()
    return fig, ax
