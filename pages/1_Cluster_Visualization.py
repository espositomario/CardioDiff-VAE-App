from utils.my_module import *

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

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
        
        
    with st.popover("Gene",icon=":material/search:"):
        #st.markdown("<h4 style='text-align: center;'>Find Cluster by Gene</h4>", unsafe_allow_html=True)

        # Use session state for text input
        gene_query = st.text_input(
            "Refseq Mouse Gene Symbol",
            placeholder="e.g., Gata4",
            value=st.session_state.gene_query,
            key="gene_query",
        )

        if gene_query:
            if gene_query in DATA.index:
                new_k = DATA.loc[gene_query, 'Cluster']
                
                CC = st.columns(2)
                with CC[0]:
                    st.write(f"{gene_query} is in Cluster {new_k}.")
                
                # Button to update the cluster and clear the gene query
                with CC[1]:
                    st.button("",icon=":material/arrow_right:", on_click=lambda: update_cluster_and_clear_query(new_k))
            else:
                st.write(f"Gene symbol: '{gene_query}' not found in the data. (Only mouse RefSeq gene symbols are accepted, notice that not all genes were included in the dataset)")

    
    download_genes_list(GENE_LIST, k, key="download_gene_list_1")

    



st.markdown("<h1 style='text-align: center;'>Clusters composition</h1>", unsafe_allow_html=True)

C = st.columns(2, gap="large")

# Cluster composition file viewers
CV_file = f"./data/plots/Clusters_CV.pdf"
Gonzalez_file = f"./data/plots/Clusters_Gonzalez.pdf"

with C[0]:
    pdf_viewer(CV_file)
    try:
        with open(CV_file, "rb") as pdf_file:
            CV_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=CV_data,
            file_name="CV_Categories_Clusters_Intersection.pdf",
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.error("File not found.")

with C[1]:
    pdf_viewer(Gonzalez_file)
    try:
        with open(Gonzalez_file, "rb") as pdf_file:
            Gonzalez_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=Gonzalez_data,
            file_name="Gonzalez_Categories_Clusters_Intersection.pdf",
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.error("File not found.")

# --------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Explore clusters</h1>", unsafe_allow_html=True)

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
        title="Cluster composition in term of categories",
        xaxis=dict(title="# of genes"),
        yaxis=dict(title="", showticklabels=False),
        plot_bgcolor="white",
        showlegend=False  # Hide the legend

    )
    # Remove duplicate legend entries (if categories repeat across features)


    return fig



C= st.columns(3)
with C[0]:
    bar_comp= plot_stacked_bar(DATA[DATA['Cluster'] == k], ["ESC_ChromState_Gonzalez2021","CV_Category"] , COLOR_DICTS)
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


#--------------------------------------------------------------
with st.expander("Cluster Data Table"):
    SEL = DATA[DATA['Cluster'] == k]
    df_tabs(SEL)
    download_genes_list(GENE_LIST, k, key="download_gene_list_2")
    



