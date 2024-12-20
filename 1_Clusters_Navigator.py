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
    

# Initialize the session state flag if not already done
if 'expand_all' not in st.session_state:
    st.session_state.expand_all = True  # Default to collapsed   
    
with st.sidebar:

    
    
    
    # Cluster selector input
    k = st.number_input(
        label="Select a Cluster (between 0 and 79)",
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
            key="gene_query_select",
        )

        # Check if a gene is selected
        if gene_query and gene_query != "":  # Ignore empty selection
            new_k = DATA.loc[gene_query, 'Cluster']
            
            update_cluster_and_clear_query(new_k)
    
    download_genes_list(GENE_LIST, k, key="download_gene_list_1")

    #
    C = st.columns(4, gap="small")
    with C[0]:
        if st.button("", icon=":material/expand:",):
            st.session_state.expand_all = True  # Set to True to expand all
    with C[1]:
        if st.button("", icon=":material/compress:"):
            st.session_state.expand_all = False  # Set to False to collapse all
#--------------------------------------------------------------
EXPANDED = False
# --------------------------------------------------------------

st.markdown("<h1 style='text-align: center;'>Cluster Navigator</h1>", unsafe_allow_html=True,)
st.markdown("<h5 style='text-align: center;'>Explore genes in each cluster in term of input features distributions, Metagene plots...</h5>", unsafe_allow_html=True,)
                

#-------------------Clusters composition-------------------#
with st.expander("Clusters categories Maps",  icon=":material/stacked_bar_chart:", expanded=st.session_state.expand_all):
    st.markdown("<h3 style='text-align: center;'>Genes distribution among clusters by categories</h3>", unsafe_allow_html=True)

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


st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(f"<h5 style='text-align: center;'>Select a cluster from the sidebar</h5>", unsafe_allow_html=True)   
st.markdown(f"<h3 style='text-align: center;'>Cluster {k} (n= {NUM_OF_GENES})</h3>", unsafe_allow_html=True)   

#------------------------------------------Features distributions------------------------------------------#

#------------------------------------------Features distributions------------------------------------------#
with st.expander("Features distributions",  icon=":material/bar_chart:",expanded=st.session_state.expand_all):
    st.markdown("<h3 style='text-align: center;'>Features distributions</h3>", unsafe_allow_html=True,help="..")

    C = st.columns(4, gap="small")

    # Create a PdfPages object to store all the plots
    pdf_path = "/tmp/plots.pdf"
    pdf_pages = PdfPages(pdf_path)

    # Loop through the features and generate the plots
    for i, feature in enumerate(['RNA', 'H3K4me3', 'H3K27ac', 'H3K27me3']):
        FILT_DF = DATA[[f"{feature}_{ct}" for ct in CT_LIST]].copy()
        VMIN, VMAX = FILT_DF.min().min(), FILT_DF.max().max()
        VMIN = FILT_DF.quantile([0.001]).min(axis=1)[0.001]
        VMAX = FILT_DF.quantile([0.999]).max(axis=1)[0.999]
        
        FILT_DF = FILT_DF[DATA['Cluster'] == k]
        
        with C[i]:
            fig, ax = plot_violin_box(FILT_DF, feature, CT_LIST, HM_COL_DICT, CT_COL_DICT, y_lab='Z-score' if i==0 else '',
                                        VMIN=VMIN, VMAX=VMAX)

            # Display the plot in the Streamlit app
            st.pyplot(fig)

            # Save the figure to the PDF
            pdf_pages.savefig(fig)  # This adds the figure to the PDF file

    # Close the PDF after adding all the plots
    pdf_pages.close()

    # Add a download button for the PDF
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="",
            icon=":material/download:",
            data=pdf_file,
            file_name=f"C{k}_Feature_distributions.pdf",
            mime="application/pdf"
        )






#------------------------------------------TSS and Categories------------------------------------------#
with st.expander("MetaGene plots",  icon=":material/area_chart:", expanded=st.session_state.expand_all):

    C= st.columns([3,1])
    with C[1]:
        st.markdown("<h3 style='text-align: center;'>Categories</h3>", unsafe_allow_html=True)
        bar_comp= plot_stacked_bar(DATA[DATA['Cluster'] == k], ["ESC_ChromState_Gonzalez2021","CV_Category"] , COLOR_DICTS)

        st.plotly_chart(bar_comp, use_container_width=True)

    with C[0]:


        # Define file paths
        tss_plot_pdf_file = f"./data/plots/TSSplots/C{k}_ext.pdf"
        ora_plot_pdf = f"./data/plots/ORA/Cluster_{k}.pdf"


        st.markdown("<h3 style='text-align: center;'>TSS Plot</h3>", unsafe_allow_html=True,
                    help="Transcription Starting Site Metaplots.\n - target histone marks on the rows\n - cell types in columns\n - dashed lines represents the control")
        pdf_viewer(tss_plot_pdf_file, )
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





#--------------------------------------------------------------
with st.expander("Expression of selected genes", icon=":material/timeline:", expanded=st.session_state.expand_all):
    st.markdown("<h3 style='text-align: center;'>Expression of selected genes </h3>", unsafe_allow_html=True)
    
    # Randomly select 16 genes as default
    
    random.seed(42)
    SEL = DATA[DATA['Cluster'] == k]

    default_genes = random.sample(SEL.index.to_list(), 18)
    
    trend_container = st.container()
    
    SEL_MODE = trend_container.segmented_control("Select genes", [ "Random","Custom"], default= 'Random',selection_mode='single', key='sel_mode_gene_trends')

    if SEL_MODE == 'Random':
            SEL_GENES = default_genes
            
    else:
            SEL_GENES = trend_container.multiselect(
                f"Selected genes ({SEL_MODE})", 
                label_visibility="collapsed",
                options=SEL.index, 
                key="select_a_gene",
                placeholder="Select genes belonging to the cluster",
            )
    
    if SEL_GENES:
        FPKM = SEL.filter(FPKM_features)
        fig = plot_gene_trend(np.log2(FPKM+1), SEL_GENES, CT_LIST, CT_COL_DICT, Y_LAB="log2(FPKM+1)")
        st.plotly_chart(fig, use_container_width=True)
        
        plotly_download_pdf(fig, file_name=f"C{k}_GeneTrends.pdf")

#--------------------------------------------------------------
with st.expander("Functional Term Enrichment Analysis", icon=":material/hdr_strong:", expanded=st.session_state.expand_all):
    st.markdown("<h3 style='text-align: center;'>Functional Term Enrichment Analysis</h3>", unsafe_allow_html=True)
    
    TOP5 = pd.read_csv("./data/TOP5_TermsPerCluster.csv", index_col=0)
    TOP5_FILT= TOP5[TOP5['Cluster']==k]
    
    if not TOP5_FILT.empty:
        # Create consistent color mapping
        unique_gene_sets = TOP5_FILT['Gene_set'].unique()
        color_dict = create_gene_set_colors(unique_gene_sets)

        # Create the plot
        fig = create_gene_set_plot(TOP5_FILT, color_dict)
        st.plotly_chart(fig, use_container_width=True)
        plotly_download_pdf(fig, file_name=f"C{k}_TermEnrichmentResults.pdf")


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
with st.expander("Information of the Cluster", icon=":material/table_rows:",expanded=st.session_state.expand_all):
    #SEL_ = SEL.drop(columns='Cluster')
    df_tabs(SEL.drop(columns='Cluster'))
    


add_footer()