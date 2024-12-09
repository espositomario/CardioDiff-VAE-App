from utils.my_module import *

#st.set_page_config(layout="wide", initial_sidebar_state="expanded", 
#                    page_icon=":material/search:",)

# Define session state variables for `k` and `gene_query`
if "k" not in st.session_state:
    st.session_state.k = 76  # Default cluster
if "gene_query" not in st.session_state:
    st.session_state.gene_query = ""

# Function to update `k` from gene query and clear `gene_query`
def update_cluster_and_clear_query(k_value):
    st.session_state.k = k_value  # Update `k` in session state
    st.session_state.gene_query = ""  # Clear the query after updating

def download_genes_list(GENE_LIST, k, key):
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
        file_name=f"C{k}_GeneIDsList.txt",
        mime="text/plain",
        key=key
    )
    
    
    
with st.sidebar:

    # Cluster selector input
    k = st.number_input(
        label="Cluster",
        min_value=0,
        max_value=79,
        step=1,
        value=st.session_state.k,
        placeholder="Enter a number between 0 and 79",
        key="cluster_input",
    )
    
    NUM_OF_GENES = GENE_CLUSTERS[str(k)]['len']
    GENE_LIST = GENE_CLUSTERS[str(k)]['gene_list']
        
    with st.popover("Gene", icon=":material/search:"):
        # Single-select dropdown for gene query
        gene_query = st.selectbox(
            "Find in which cluster a gene is",
            options=[""] + list(DATA.index),  # Empty string for no default selection
            format_func=lambda x: "Type a gene symbol..." if x == "" else x,
            help='Refseq Gene Symbol in Mouse',
            key="gene_query",
        )

        # Check if a gene is selected
        if gene_query and gene_query != "":  # Ignore empty selection
            new_k = DATA.loc[gene_query, 'Cluster']
            
            CC = st.columns(2, vertical_alignment="center")
            with CC[0]:
                st.write(f"Go to Cluster {new_k}")
            
            # Button to update the cluster and clear the gene query
            with CC[1]:
                st.button(":material/keyboard_arrow_right:", on_click=lambda: update_cluster_and_clear_query(new_k))
    
    download_genes_list(GENE_LIST, k, key="download_gene_list_1")

#--------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Explore clusters</h1>", unsafe_allow_html=True)


with st.expander("Clusters categories Maps", expanded=True):
    #st.markdown("<h1 style='text-align: center;'>Clusters composition</h1>", unsafe_allow_html=True)

    C = st.columns(2, gap="large")

    # Cluster composition file viewers
    CV_file = f"./data/plots/Clusters_CV.pdf"
    Gonzalez_file = f"./data/plots/Clusters_Gonzalez.pdf"

    with C[0]:
        pdf_viewer(CV_file, key="CV_pdf")
        try:
            with open(CV_file, "rb") as CV_file_pdf:
                CV_data = CV_file_pdf.read()
            st.download_button(
                label="",
                key="download_CV",
                icon=":material/download:",
                data=CV_data,
                file_name="CV_Categories_Clusters_Intersection.pdf",
                mime="application/pdf",
            )
        except FileNotFoundError:
            st.error("File not found.")

    with C[1]:
        pdf_viewer(Gonzalez_file,  key="Gonzalez_pdf")
        try:
            with open(Gonzalez_file, "rb") as Gonzalez_file_pdf:
                Gonzalez_data = Gonzalez_file_pdf.read()
            st.download_button(
                label="",
                key="download_Gonzalez",
                icon=":material/download:",
                data=Gonzalez_data,
                file_name="Gonzalez_Categories_Clusters_Intersection.pdf",
                mime="application/pdf",
            )
        except FileNotFoundError:
            st.error("File not found.")

# --------------------------------------------------------------


# Create layout for exploring clusters
CC = st.columns(6, vertical_alignment="bottom", gap="large")




#--------------------------------------------------------------            
st.markdown(f"<h3 style='text-align: center;'>Cluster {k} (n= {NUM_OF_GENES})</h3>", unsafe_allow_html=True)   



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
        yaxis=dict(title="", showticklabels=False),
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



C= st.columns([1,3])
with C[0]:
    bar_comp= plot_stacked_bar(DATA[DATA['Cluster'] == k], ["ESC_ChromState_Gonzalez2021","CV_Category"] , COLOR_DICTS)
    
    #plot_frame()
    
    
    st.plotly_chart(bar_comp, use_container_width=True)








C = st.columns(2, gap="large")
with C[0]:    
    
    # Define file paths
    tss_plot_pdf_file = f"./data/plots/TSSplots/C{k}_ext.pdf"
    ora_plot_pdf = f"./data/plots/ORA/Cluster_{k}.pdf"


    st.markdown("<h3 style='text-align: center;'>TSS Plot</h3>", unsafe_allow_html=True,
                help="Transcription Starting Site Metaplots.\n - target histone marks on the rows\n - cell types in columns\n - dashed lines represents the control")
    pdf_viewer(tss_plot_pdf_file)
    try:
        with open(tss_plot_pdf_file, "rb") as pdf_file:
            tss_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=tss_data,
            file_name=f"C{k}_TSSPlot.pdf",
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.error("TSS plot file not found.")


with C[1]:

    st.markdown("<h3 style='text-align: center;'>Term Enrichment</h3>", unsafe_allow_html=True)
    if os.path.exists(ora_plot_pdf):
        pdf_viewer(ora_plot_pdf)
        with open(ora_plot_pdf, "rb") as pdf_file:
            ora_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=ora_data,
            file_name=f"C{k}_TermEnrichment.pdf",
            mime="application/pdf",
        )
    else:
        
        st.markdown(
        """
        <div style="text-align: center; font-size: 16px;">
            No significant term resulted (adj. p-value > 0.05)
        </div>
        """, 
        unsafe_allow_html=True
        )


SEL = DATA[DATA['Cluster'] == k]




import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    
#--------------------------------------------------------------
with st.expander("Gene Expression Dynamics", icon=":material/timeline:", expanded=True):
    st.markdown("<h3 style='text-align: center;'>Gene expression dynamics across differentiation</h3>", unsafe_allow_html=True)
    
    # Randomly select 16 genes as default
    
    random.seed(42)
    default_genes = random.sample(SEL.index.to_list(), 18)
    
    trend_container = st.container()
    
    
    
    SEL_MODE = trend_container.segmented_control("Selection mode", [ "Random","Custom"], default= 'Random',selection_mode='single', key='sel_mode_gene_trends')


    if SEL_MODE == 'Random':
            SEL_GENES = default_genes
            
    else:
            SEL_GENES = trend_container.multiselect(
                f"Selected genes ({SEL_MODE})", 
                options=SEL.index, 
                key="select_a_gene",
                placeholder="Select genes in the cluster...",
            )
    
    if SEL_GENES:
        FPKM = SEL.filter(FPKM_features)
        fig = plot_gene_trend(np.log2(FPKM+1), SEL_GENES, CT_LIST, CT_COL_DICT, Y_LAB="log2(FPKM+1)")
        st.plotly_chart(fig, use_container_width=True)



#--------------------------------------------------------------
with st.expander("Cluster Data Table", icon=":material/table_rows:"):
    df_tabs(SEL)
    download_genes_list(GENE_LIST, k, key="download_gene_list_2")
    