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
import io
import plotly.io as pio
#custom components
from streamlit_pdf_viewer import pdf_viewer
from streamlit_extras.bottom_container import bottom
from plotly.subplots import make_subplots
import plotly.colors as pc
import colorsys

import fitz  # PyMuPDF
from PIL import Image

import math
from matplotlib import cm, colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import gseapy as gp

HOME_LINK = "https://cardiodiff-vae.streamlit.app"
APP_NAME= 'CardioDiff-VAE'


PAGES_DICT = {
    1: {'file': './1_Latent_Space.py', 'name': 'Gene search', 'image': './data/plots/page1.jpg', 
        'description': 'Explore the genome in the VAE latent space.'},
    
    3: {'file': './3_Clusters_Navigator.py', 'name': 'Clusters', 'image': './data/plots/page2.jpg',
        'description': 'Explore the clusters in the VAE latent space.'},
    
    2: {'file': './2_Ontology_Navigator.py', 'name': 'Term search', 'image': './data/plots/page3.jpg',
        'description': 'Explore the ontologies in the VAE latent space.'},
    
    4: {'file': './4_Data.py', 'name': 'Data', 'image': './data/plots/page4.jpg', 
        'description': 'Explore the data.'},
}

ABOUT_DICT = {
    1:{'name': 'Mario Esposito', 'image': './data/plots/Mario_Esposito.jpg', 
    'description': 'Mario Esposito obtained his Bachelor’s degree in Health Biotechnology from the University of Naples Federico II in 2022. During his undergraduate studies, he interned at CEINGE Biotecnologie avanzate in Naples in the Mario Capasso Lab. \
            In 2024, he completed his Master’s degree in Bioinformatics at the University of Bologna. During his Master’s studies, he joined the Centre for Genomic Regulation (CRG) in Barcelona as an Erasmus+ intern in the Luciano Di Croce Lab (2024)'},
    
    2:{'name': 'Luciano Di Croce', 'image': './data/plots/Luciano_Di_Croce.jpg',
    'description': 'text'},
    3:{'name': 'Enrique Blanco', 'image': './data/plots/Enrique_Blanco.jpg',
    'description': 'text'},
}

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

CV_COL_DICT= {'other':'#ECECEC',
                'RNA_ESC': '#405074',
                'RNA_MES': '#7d5185',
                'RNA_CP': '#c36171',
                'RNA_CM': '#eea98d',
                'STABLE':'#B4CD70',
                }
GONZALEZ_COL_DICT= {'other':'#ECECEC','Active': '#E5AA44','Bivalent': '#7442BE'}

COLOR_DICTS = {
    "CV_Category": CV_COL_DICT,
    "ESC_ChromState_Gonzalez2021": GONZALEZ_COL_DICT,
}


def page_cover(PAGE_DICTS, i):
    st.image(PAGE_DICTS[i]['image'], use_container_width=True)
    if st.button(PAGE_DICTS[i]['name'], key=PAGE_DICTS[i]['name'], use_container_width=True):
        st.switch_page(PAGE_DICTS[i]['file'])
        
        
def style_dataframe(df, header_bg_color="#4CAF50", header_text_color="#FFFFFF"):
    """
    Style a DataFrame for display in Streamlit with custom header background and text colors.

    Parameters:
    df (pd.DataFrame): The DataFrame to style.
    header_bg_color (str): Background color for the header (default: "#4CAF50").
    header_text_color (str): Text color for the header (default: "#FFFFFF").

    Returns:
    pandas.io.formats.style.Styler: Styled DataFrame.
    """
    # Increase the Pandas Styler limit to handle large DataFrames
    pd.set_option("styler.render.max_elements", df.size)
    # Apply custom styles to the DataFrame
    styled_df = df.style.set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', header_bg_color),
                ('color', header_text_color),
                ('font-weight', 'bold'),
                ('text-align', 'center')
            ]
        },
        {
            'selector': 'thead',
            'props': [('border-bottom', '1px solid black')]
        },
        {
            'selector': 'tbody tr',
            'props': [('text-align', 'center')]
        }
    ])
    return styled_df


def style_dataframe(df, header_bg_color='lightblue', header_text_color='black'):
    """
    Styles a DataFrame for display in Streamlit, customizing the header background and text colors.

    Args:
        df: The DataFrame to style.
        header_bg_color: The background color for the header.
        header_text_color: The text color for the header.

    Returns:
        The styled DataFrame.
    """
    pd.set_option("styler.render.max_elements", df.size)

    styled_df = df.style.set_properties(**{'background-color': 'white', 'color': 'black'})
    styled_df = styled_df.set_table_styles([
        {'selector': 'th',
         'props': [
             ('background-color', header_bg_color),
             ('color', header_text_color),
             ('font-weight', 'bold')
         ]}
    ])
    return styled_df




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

#https://www.ncbi.nlm.nih.gov/gene/?term=(Mus+musculus%5BOrganism%5D)+AND+{GENE}%5BGene+Name%5D
    st.dataframe(
    df,
    column_config={
        "NCBI": st.column_config.LinkColumn(
            "NCBI",
            help="Click to open the NCBI Gene page",
            #validate=r"^https://[a-z]+\.streamlit\.app$",
            #max_chars=100,
            display_text=r"AND\+(.+?)%5BGene\+Name%5D"

            #display_text="NCBI-Gene"
        ),
    },
    #hide_index=True,
    )

    return df

def df_tabs(DF):
    # Feature groups
    Z_AVG =  DF[['NCBI']+[col for col in Z_AVG_features if col in DF.columns]]
    LOG_FC = DF[['NCBI']+[col for col in LOG_FC_features if col in DF.columns]]
    MISC =   DF[['NCBI']+[col for col in MISC_features if col in DF.columns]]
    LATENT = DF[['NCBI']+[col for col in LATENT_features if col in DF.columns]]
    FPKM =   DF[['NCBI']+[col for col in FPKM_features if col in DF.columns]]

    # Create Streamlit tabs
    tabs = st.tabs(["Full table", "Features Z-scores", "RNA Log FCs", "Annotations", "VAE Latent Space", "RNA FPKMs"])

    with tabs[0]:  # All
        st.markdown("### Full table", )
        filter_dataframe(DF)

    with tabs[1]:  # Z-scores
        st.markdown("### Input features Z-scores", )
        filter_dataframe(Z_AVG)

    with tabs[2]:  # Log Fold Changes
        st.markdown("### Input RNA Log Fold Changes (LogFCs)", )
        filter_dataframe(LOG_FC)

    with tabs[3]:  # Annotations
        st.markdown("### Gene annotations", )
        filter_dataframe(MISC)

    with tabs[4]:  # VAE Latent Space
        st.markdown("### VAE Latent Variables and UMAP projections", )
        filter_dataframe(LATENT)

    with tabs[5]:  # RNA FPKMs
        st.markdown("### RNA-seq FPKMs per replicate", )
        filter_dataframe(FPKM)





# Load gene cluster dictionary
with open(f'./data/gene_clusters_dict.pkl', 'rb') as f:
    GENE_CLUSTERS = pickle.load(f)

# Load DATA
DATA = pd.read_csv(f'./data/DATA.csv', index_col='GENE')
DATA['Cluster'] = pd.Categorical(DATA['Cluster'])

NCBI_col = DATA.index.to_series().apply(lambda GENE: f"https://www.ncbi.nlm.nih.gov/gene/?term=(Mus+musculus%5BOrganism%5D)+AND+{GENE}%5BGene+Name%5D")

# Create the NCBI column and insert it as the first column
DATA.insert(
    0,  # Position (0 means first column)
    "NCBI",  # Column name
    NCBI_col  # Column data
)
#
RNA_FPKM= pd.read_csv(f'./data/RNA_FPKMs.csv', index_col='GENE')

#
NCBI_IDs = pd.read_csv('./data/NCBI_ID.csv', index_col=0)

# List of continuous features for coloring
continuous_features = ["RNA_CV", "VAE_RMSE", "VAE_Sc"]


Z_AVG_features = ['RNA_ESC', 'RNA_MES', 'RNA_CP', 'RNA_CM', 'H3K4me3_ESC', 'H3K4me3_MES',
        'H3K4me3_CP', 'H3K4me3_CM', 'H3K27ac_ESC', 'H3K27ac_MES', 'H3K27ac_CP',
        'H3K27ac_CM', 'H3K27me3_ESC', 'H3K27me3_MES', 'H3K27me3_CP',
        'H3K27me3_CM']


LOG_FC_features = ['RNA_MES_ESC_FC', 'RNA_CP_ESC_FC','RNA_CM_ESC_FC','RNA_CP_MES_FC','RNA_CM_MES_FC', 'RNA_CM_CP_FC', ]

MISC_features = [ 'Cluster','RNA_CV', 'CV_Category', 'ESC_ChromState_Gonzalez2021', 'VAE_RMSE', 'VAE_Sc']

LATENT_features = ['VAE1', 'VAE2', 'VAE3', 'VAE4', 'VAE5', 'VAE6', 'VAE_UMAP1', 'VAE_UMAP2', 'VAE_PCA1(43%)', 'VAE_PCA2(18%)']

FPKM_features = [ 'RNA_ESC_1', 'RNA_ESC_2', 'RNA_MES_1', 'RNA_MES_2',
            'RNA_CP_1', 'RNA_CP_2', 'RNA_CM_1', 'RNA_CM_2']


COLOR_FEATURES =  Z_AVG_features + LOG_FC_features + MISC_features





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
        label="Data (12MB)",
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
        data=melted_df, x='CT', y='Value', palette=[ct_col_dict[ct] for ct in ct_list], hue='CT',
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
        data=melted_df, x='CT', y='Value', palette=[ct_col_dict[ct] for ct in ct_list], hue='CT',
        ax=ax, jitter=True, size=2, alpha=0.5, linewidth=0.5, 
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



def plot_stacked_bar(DATA, feature_columns, COLOR_DICTS):
    """
    Plot stacked bar charts for multiple features in a single plot.

    Parameters:
    - DATA: pandas DataFrame containing the data.
    - feature_columns: List of column names to plot.
    - COLOR_DICTS: Dictionary where keys are column names and values are color dictionaries.
    """
    fig = go.Figure()

    for col in feature_columns:
        # Get value counts and corresponding colors
        counts = DATA[col].value_counts().to_dict()
        color_dict = COLOR_DICTS.get(col, {})  # Default to empty if no dict provided

        # Sort counts by color_dict key order
        sorted_categories = [key for key in color_dict.keys() if key in counts]
        sorted_counts = {cat: counts[cat] for cat in sorted_categories}

        # Add bars for each category in the column
        for category in sorted_categories:
            count = sorted_counts.get(category, 0)
            fig.add_trace(
                go.Bar(
                    x=[count],  # Counts on x-axis
                    y=[col],  # Feature name on y-axis
                    orientation='h',  # Horizontal bar orientation
                    marker_color=color_dict.get(category, color_dict.get("other", "grey")),
                    hovertemplate=f"{category}: {count}<extra></extra>",  # Display category and count

                    text=[category],  # Display the category name
                    textposition="inside",  # Position text horizontally inside the bar
                )
            )
    # Update layout for stacked bar
    fig.update_layout(
        barmode="stack",
        title=None,  # Remove the title
        margin=dict(t=0),  # Remove the top margin to eliminate space
        xaxis=dict(title="# of genes"),
        yaxis=dict(title="", showticklabels=True, tickvals=feature_columns, ticktext=['Chromatin State in ESC', 'Expression Peaks/Stable',]),
        plot_bgcolor="white",
        showlegend=False,  # Hide the legend
        height=250,  # Set a fixed height
        width=400,  # Set a fixed width

    )
    # Remove duplicate legend entries (if categories repeat across features)


    return fig




def plot_frame(border_color="#FFFFFF"):
    PLOT_BGCOLOR = border_color

    st.markdown(
        f"""
        <style>
        .stPlotlyChart {{
        outline: 1px solid {PLOT_BGCOLOR};
        border-radius: 4px;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.20), 0 3px 10px 0 rgba(0, 0, 0, 0.30);
        }}
        </style>
        """, unsafe_allow_html=True
)


def plot_gene_trend(DATA, SEL_GENES, CT_LIST, CT_COL_DICT, Y_LAB):
    """
    Plot trends for selected genes across conditions with arrow connections between average points.
    
    Parameters:
        DATA (pd.DataFrame): DataFrame where rows are genes, columns are RNA counts.
        SEL_GENES (list): List of gene names to plot.
        CT_LIST (list): List of ordered conditions (e.g., ["ESC", "MES", "CP", "CM"]).
        CT_COL_DICT (dict): Mapping of condition names to colors.
        Y_LAB (str): Label for the Y-axis.

    Returns:
        Plotly figure with subplots for each gene's trend.
    """
    num_genes = len(SEL_GENES)
    num_cols = 6  # Fixed number of columns
    num_rows = math.ceil(num_genes / num_cols)  # Calculate number of rows based on the number of genes

    # Create subplot figure
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=SEL_GENES,
        horizontal_spacing=0.05, vertical_spacing=0.1,
    )

    for i, gene_name in enumerate(SEL_GENES):
        # Extract gene data
        gene_data = DATA.loc[gene_name]

        # Extract condition (CT) and replicate (REP) information
        CT = pd.Categorical(gene_data.index.str.extract(f"({'|'.join(CT_LIST)})")[0], categories=CT_LIST, ordered=True)
        REP = gene_data.index.str.extract(r'(\d)')[0]
        df = pd.DataFrame({Y_LAB: gene_data.values, 'CT': CT, 'REP': REP})

        # Filter out invalid rows
        df = df.dropna()

        # Compute average values for arrows
        avg_df = df.groupby('CT', observed=False)[Y_LAB].mean().reset_index().sort_values('CT')

        # Determine subplot location
        row = (i // num_cols) + 1
        col = (i % num_cols) + 1

        # Add scatter plot for individual points
        for ct in CT_LIST:
            ct_data = df[df['CT'] == ct]
            fig.add_trace(
                go.Scatter(
                    x=ct_data['CT'],
                    y=ct_data[Y_LAB],
                    mode='markers',
                    marker=dict(size=12, color=CT_COL_DICT[ct]),
                    name=ct,
                    showlegend=False,
                    hovertemplate=f"{ct}"
                ),
                row=row, col=col
            )

        # Add arrows between average points
        for j in range(len(avg_df) - 1):
            fig.add_annotation(
                x=avg_df.iloc[j + 1]["CT"],
                y=avg_df.iloc[j + 1][Y_LAB],
                ax=avg_df.iloc[j]["CT"],
                ay=avg_df.iloc[j][Y_LAB],
                xref=f"x{i + 1}",
                yref=f"y{i + 1}",
                axref=f"x{i + 1}",
                ayref=f"y{i + 1}",
                arrowhead=5,
                arrowsize=2,
                arrowwidth=1,
                arrowcolor="grey",
                showarrow=True,
            )

        # Update subplot axes
        y_max = math.ceil(df[Y_LAB].max() * 1.1)  # Determine max value with padding
        fig.update_xaxes(title_text="", row=row, col=col, showticklabels=False)
        fig.update_yaxes(
            title_text=Y_LAB if col == 1 else "",
            row=row, col=col,
            range=[0, y_max],  # Limit range from 0 to max
            tickvals=[0, y_max],  # Display only 0 and max
            ticktext=[0, y_max],
        )

        # Add a horizontal line at y_max
        fig.add_shape(
            type="line",
            x0=0,
            x1=len(CT_LIST),  # Covers the entire x-axis range
            y0=y_max,
            y1=y_max,
            xref=f"x{i + 1}",
            yref=f"y{i + 1}",
            line=dict(color="grey", width=0.5),
            row=row,
            col=col,
        )

    # Update figure layout
    fig.update_layout(
        height=100+(200 * num_rows), width=300 * num_cols,
        title_text=None,
        showlegend=False,
        plot_bgcolor="white",
    )

    return fig


def add_footer():
    """
    Adds a footer to the Streamlit app with the user's name, copyright notice,
    and clickable GitHub and LinkedIn icons with proper inline size adjustments.
    """
    footer_content = """
        <style>
        .footer {
            margin-top: 50px;
            margin-bottom: 10px;
            padding: 30px 0;
            text-align: center;
            font-size: 14px;
            color: #7d7c7c;
            border-top: 1px solid #ddd;
        }
        .footer a {
            color: #0366d6;
            text-decoration: none;
            margin: 0 5px;
        }
        </style>
        <div class="footer">
            <p>Mario Esposito, Luciano Di Croce and Enrique Blanco © 2025 | 
                <a href="https://github.com/espositomario" target="_blank">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width: 20px; height: 20px; vertical-align: text-center;" />
                </a>
            </p>
        </div>
    """
    st.markdown(footer_content, unsafe_allow_html=True)
    


def plotly_download_pdf(fig, file_name):
    """
    Generates a PDF from a Plotly figure and adds a download button in Streamlit.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to save to PDF.
        pdf_path (str): Path where the PDF will be saved (default is "/tmp/plot.pdf").
        button_label (str): Label for the download button (default is 'Download PDF').
    """
    # Make sure the output directory exists
    pdf_path = f"/tmp/{file_name}"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Save the Plotly figure to a PDF file
    pio.write_image(fig, pdf_path, format='pdf')
    # Add a download button for the generated PDF
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="",
            icon=":material/download:",
            data=pdf_file,
            file_name=file_name,
            mime="application/pdf",
            key=f"download_{file_name}"
        )

def plotly_download_png(fig, file_name, scale=2):
    """
    Generates a PNG from a Plotly figure and adds a download button in Streamlit.

    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to save to PNG.
        file_name (str): Name of the PNG file (without extension).
        scale (int, optional): Scaling factor for the image resolution. Defaults to 2.
    """
    png_path = f"/tmp/{file_name}.png"  # Add .png extension
    os.makedirs(os.path.dirname(png_path), exist_ok=True)

    try:
        pio.write_image(fig, png_path, format="png", scale=scale)

        key = np.random.randint(0, 1000000)
        
        with open(png_path, "rb") as png_file:
            st.download_button(
                label="",
                icon=":material/download:",
                data=png_file,
                file_name=f"{file_name}.png",  # Include extension in download name
                mime="image/png",
                key=f"download_{file_name}{key}_png"  # Unique key
            )
    except Exception as e:
        st.error(f"An error occurred during PNG export: {e}") #handle exceptions
        return None

def create_gene_set_colors(gene_sets):
    """
    Create a dictionary mapping each gene set to a unique color.

    Args:
        gene_sets (list): List of unique gene sets.

    Returns:
        dict: Mapping of gene sets to colors.
    """
    colors = pc.qualitative.Set1  # Use a qualitative color scale
    color_dict = {gene_set: colors[i % len(colors)] for i, gene_set in enumerate(sorted(gene_sets))}
    return color_dict


def create_gene_set_plot(df, color_dict):
    """
    Creates a scatter plot with all terms, colored by gene set, and sized by count.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        color_dict (dict): A dictionary mapping gene sets to colors.

    Returns:
        plotly.Figure: The figure object containing the plot.
    """
    # Sort the DataFrame by gene set and reverse-sort by p-value
    df = df.sort_values(['Gene_set', '-log10(adj p-value)'], ascending=[True, False])

    # Create the plot
    fig = px.scatter(
        df,
        x='-log10(adj p-value)',
        y='Term',
        color='Gene_set',
        size='Count',
        hover_name='Term',  # Set Term as the hover title
        hover_data={
            'Odds Ratio': True,  # Include Odds Ratio
            'Genes': True,       # Include Genes
            'Count': True,       # Include Count
            'Gene_set': False,   # Exclude Gene_set from hover
        },
        title=None,
        category_orders={
            'Term': df['Term'].tolist(),  # Match the descending order on y-axis
        },
        color_discrete_map=color_dict  # Use the color dictionary for consistent coloring
    )
    # Customize hovertemplate to remove extra hover box
    fig.update_traces(
        hovertemplate='<b>Term:</b> %{hovertext}<br>'
                        '<b>Odds Ratio:</b> %{customdata[0]}<br>'
                        '<b>Genes:</b> %{customdata[1]}<br>'
                        '<b>Count:</b> %{customdata[2]}<extra></extra>'
    )


    # Add horizontal lines representing -log10(adj p-value) for each term
    for _, row in df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            x1=row['-log10(adj p-value)'],
            y0=row['Term'],
            y1=row['Term'],
            line=dict(color=color_dict[row['Gene_set']], width=1,)
        )
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',  # Set background to white
        margin=dict(l=600, r=50, b=50, t=50),
        xaxis=dict(
            gridcolor='lightgrey',  # Grey grid lines
            title='-log10(adj p-value)'
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            title=None
        ),
        xaxis_title='-log10(adj p-value)',
        yaxis_title=None,
        legend_title='Gene Set',
        width=1200,  # Fixed
        height=(len(df['Term']) * 30)+200,  # Dynamic height adjustment based on the number of terms
    )

    return fig

def title_with_help(TITLE, help_text, help_width=0.1):
    """Mostra contenuto con un popover di aiuto usando st.popover.

    Args:
        content: Elementi di Streamlit da mostrare.
        help_text: Il testo da mostrare nel popover.
        help_width: Larghezza della colonna di aiuto (relativa alle altre colonne).
    """

    C = st.columns([help_width,1, help_width], vertical_alignment="center")
    with C[1]:
        st.markdown(f"<h3 style='text-align: center;'>{TITLE}</h3>", unsafe_allow_html=True)
    with C[2]:
        with st.popover("", icon=':material/help:'):  # Usa st.popover
            st.write(help_text)
            


def convert_pdf_to_image(pdf_path, dpi=600):  # Add dpi argument with default value
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]  # Get the first (and only) page
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))  # Set DPI using matrix
        img = Image.open(io.BytesIO(pix.tobytes()))
        return img
    except fitz.fitz.FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None
    

def hum2mouse(GENE_LIST):
    new_genes = []
    for gene in GENE_LIST:
        new_gene = gene[0] + gene[1:].lower()
        new_genes.append(new_gene)
    return new_genes


def download_genes_list(GENE_LIST, k=None, key=None, filename=None):
    """
    Download a list of genes as a text file.

    Parameters:
    - GENE_LIST: List of gene IDs to download.
    - k: Cluster number for file name.
    """
    # Convert the list to a string with each element on a new line
    GENE_LIST_FILE = "\n".join(GENE_LIST)

    st.download_button(
        label="Gene List",
        icon=":material/download:",
        data=GENE_LIST_FILE,
        file_name=f"C{k}_GeneIDsList.txt" if filename is None else filename,
        mime="text/plain",
        key=key
    )
    
    



def get_gene_ncbi_page(NCBI_IDs):
    with st.popover("NCBI Gene Info"):
        GENE = st.selectbox("Get a link to the NCBI Gene entry", 
                                options=[""] + list(DATA.index),  # Empty string for no default selection
                                format_func=lambda x: "Type a gene symbol..." if x == "" else x,
                                help='Refseq Gene Symbol in Mouse',)
        if GENE:
            
            NCBI_ID = NCBI_IDs.loc[GENE]['NCBI GeneID']
            if NCBI_ID == 0:
                #st.markdown(f"[NCBI link](https://www.ncbi.nlm.nih.gov/gene/?term={GENE})", unsafe_allow_html=True)
                st.markdown(f"[NCBI link](https://www.ncbi.nlm.nih.gov/gene/?term=(Mus+musculus%5BOrganism%5D)+AND+{GENE}%5BGene+Name%5D)",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"[NCBI link](https://www.ncbi.nlm.nih.gov/gene/{NCBI_ID})", unsafe_allow_html=True)



def scatter(DATA, COLOR_FEATURES, SEL_GENES, DR, key, COLOR_DICTS=None, default_index=0,
            LABELS=True, SEL_GENES_SIZE=16, LABEL_SIZE=12, DEF_POINT_ALPHA=0.8, SEL_POINT_ALPHA=0.9):
    
    COMPONENTS = [col for col in DATA.columns if f'{DR}' in col]
    # Layout for control elements
    C = st.columns([5, 1, 1, 1], vertical_alignment="bottom", gap="small")
    with C[0]:
        # Dropdown for feature selection
        selected_feature = st.selectbox(
            "Color By", key=key + 'sel_box',
            options=COLOR_FEATURES,
            index=default_index  # Default to the first feature
        )

    CAT = False
    if pd.api.types.is_categorical_dtype(DATA[selected_feature]) or DATA[selected_feature].dtype == 'object':
        CAT = True
        color_dict = COLOR_DICTS.get(selected_feature, None)

        if color_dict is None:

            categories = DATA[selected_feature].unique()
            num_categories = len(categories)

            hsv_colors = [colorsys.hsv_to_rgb(i / num_categories, 1, 1) for i in range(num_categories)]
            hex_colors = [mcolors.to_hex(rgb) for rgb in hsv_colors]
            color_dict = {cat: hex_colors[i] for i, cat in enumerate(categories)}

    if not CAT:
        with C[1]:
            with st.popover("", icon=":material/filter_list:"):
                # Range selection slider
                min_val = DATA[selected_feature].min()
                max_val = DATA[selected_feature].max()
                selected_range = st.slider(
                    f"Range Mask on {selected_feature}", min_value=min_val, max_value=max_val,
                    value=(min_val, max_val), key=key + 'range_slider'
                )

    else:
        with C[1]:
            with st.popover("", icon=":material/filter_list:"):
                categories = sorted(DATA[selected_feature].unique().tolist())
                selected_categories = st.multiselect(
                    f"Select categories to color for {selected_feature}",
                    options=categories,
                    default=categories,  # Select all by default
                    key=key + 'cat_filter'
                )
                
                
                # Replace unselected values with 'Other'
                # sort modfiied data accordint to color_dict.keys() if modified_data[selected_feature] is not integer
                                
                modified_data = DATA.copy()
                
                if selected_feature != 'Cluster':
                    sort_order = list(color_dict.keys())  # Use color_dict keys for sorting
                    modified_data[selected_feature] = pd.Categorical(
                        modified_data[selected_feature], categories=sort_order, ordered=True
                    )
                    modified_data = modified_data.sort_values(by=selected_feature)
                else:
                    modified_data = modified_data.sort_values(by=selected_feature)
                    

                modified_data[selected_feature] = modified_data[selected_feature].apply(
                    lambda x: x if x in selected_categories else 'other')


                # Update color_dict to include 'other'
                color_dict['other'] = "#E7E7E7" # Set color for 'Other'
                

    with C[2]:
        with st.popover("", icon=":material/settings:"):
            point_size = st.slider("Point Size", min_value=1, max_value=14, value=4, step=1, key=key + 'point_size')
            point_opacity = st.slider("Transparency", min_value=0.1, max_value=1.0, value=DEF_POINT_ALPHA, step=0.1, key=key + 'point_opacity')
            if not CAT:
                colormap = st.segmented_control(
                    "Select Colormap", ["Spectral_r", "Blues", "viridis", "RdBu_r"],
                    default='Spectral_r', selection_mode='single', key=key + 'colormap'
                )
                min_col, max_col = st.slider(
                    "Color Range (percentile)",
                    min_value=1, max_value=99, value=(1, 99), step=1, key=key + 'min_max'
                )

    # Precompute colors for all points
    if CAT:
        # Categorical coloring
        colors = modified_data[selected_feature].map(
            lambda val: color_dict[val] if color_dict and val in color_dict else "gray"
        ).tolist()
    else:
            
        # Compute min and max percentiles
        min_p, max_p = np.percentile(DATA[selected_feature], [min_col, max_col])
        # Clamp values between min_p and max_p, normalize, and map to the colormap
        feature_values = DATA[selected_feature].values
        clamped_values = np.clip(feature_values, min_p, max_p)
        normalized_values = (clamped_values - min_p) / (max_p - min_p)
        # Map normalized values to colors in one operation
        cmap = cm.get_cmap(colormap)
        # Map normalized values directly to RGBA using matplotlib
        rgba_colors = cmap(normalized_values)
        # Convert RGBA to HEX for compatibility with Plotly
        colors = [mcolors.to_hex(rgba) for rgba in rgba_colors]
        # Apply range selection (masking)
        mask = (DATA[selected_feature] < selected_range[0]) | (DATA[selected_feature] > selected_range[1])
        colors = np.where(mask, "lightgrey", colors).tolist() #Mask the colors

    if CAT:
        fig = px.scatter(
            modified_data,
            x=COMPONENTS[0],  # Use dynamic component names
            y=COMPONENTS[1],
            color=selected_feature,
            color_discrete_map=color_dict,
            hover_data=[modified_data.index,  selected_feature],
            title=f"{selected_feature}",
            labels={COMPONENTS[0]: COMPONENTS[0].replace("VAE_", ""), COMPONENTS[1]: COMPONENTS[1].replace("VAE_", "")}, # Dynamic labels
        )

        fig.update_traces(marker=dict(size=point_size, opacity=point_opacity),)    
        fig.update_layout(legend=dict(
                itemsizing='constant',
                tracegroupgap=0,
                traceorder='normal',
                font=dict(
                    size=12,
                    color="black"
                ),
            ),
            )
    
    
    else:
        fig = px.scatter(
            DATA,
            x=COMPONENTS[0],  # Use dynamic component names
            y=COMPONENTS[1],
            color=DATA[selected_feature],
            color_continuous_scale=colormap,
            range_color=[min_p, max_p],
            hover_data=[DATA.index,  selected_feature],
            title=f"{selected_feature}",
            labels={COMPONENTS[0]: COMPONENTS[0].replace("VAE_", ""), COMPONENTS[1]: COMPONENTS[1].replace("VAE_", "")}, # Dynamic labels
        )
        mask = (DATA[selected_feature] < selected_range[0]) | (DATA[selected_feature] > selected_range[1])
        fig.update_traces(
            marker=dict(
                size=point_size,
                opacity=np.where(mask, 0.1, point_opacity),
            ),
        )

    # Highlight selected genes (modified)
    if SEL_GENES:
        for gene in SEL_GENES:
            if gene in DATA.index:
                gene_data = DATA.loc[gene]
                gene_color = colors[DATA.index.get_loc(gene)]
                fig.add_trace(go.Scattergl(
                    x=[gene_data[COMPONENTS[0]]],  # Use dynamic component names
                    y=[gene_data[COMPONENTS[1]]],
                    mode="markers",
                    marker=dict(size=SEL_GENES_SIZE, color=gene_color, line=dict(width=1, color="black"), opacity=SEL_POINT_ALPHA),
                    showlegend=False,  # Avoid duplicate legend entries
                    hovertemplate=(
                        f"<b>Gene:</b> {gene}<br>"
                        f"<b>{COMPONENTS[0]}:</b> {gene_data[COMPONENTS[0]]}<br>" # Dynamic hover data
                        f"<b>{COMPONENTS[1]}:</b> {gene_data[COMPONENTS[1]]}<br>"
                        f"<b>{selected_feature}:</b> {gene_data[selected_feature]}<br>"
                        "<extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor="red", font=dict(color="white"))  # Red hover background
                ))

                if LABELS:
                    fig.add_annotation(
                        x=gene_data[COMPONENTS[0]],
                        y=gene_data[COMPONENTS[1]],
                        text=gene,
                        #showarrow=True,
                        arrowsize=0.5,
                        arrowcolor="black",
                        arrowhead=0,
                        font=dict(size=LABEL_SIZE, color="black"),  # Text font
                        bgcolor="white",  # Background color for the annotation
                        #bordercolor="black",  # Optional: Add border to the annotation
                        borderwidth=0,
                        borderpad=0,
                    
                    )

    # Customize layout
    fig.update_layout(
        title='',
        margin=dict(t=50, r=50, b=50, l=50),
        xaxis=dict(showgrid=False, zeroline=False,  tickvals=[]),
        yaxis=dict(showgrid=False, zeroline=False,   tickvals=[]),
        plot_bgcolor="white",
        autosize=False,
        #showlegend=True
    )

    with C[3]:
        plotly_download_png(fig, file_name=f"VAE_LatentSpace_{DR}_{selected_feature}")
    
    return fig
