from utils.my_module import *

#st.set_page_config(layout="wide", initial_sidebar_state="expanded", 
#                    page_icon=":material/search:",)

HELP_DICT = {
    "Cluster categories Map": "- On the LEFT: Tree maps depicting the percentage of genes in each cluster defined as most variable\
        and most stable in term of expression coefficient of variation . Five lists were generated: the most variable genes (top 4,000 CV) stratified by their\
        CTmax values (ESC, MES, CP, CM) and the most stable genes (bottom 4,000 CV).\
            \n- On the RIGHT: Tree maps depicting the percentage of genes in each cluster annotated with an active,\
            bivalent or other chromatin state (in ESC) in another study (Gonzalez, 2021).\
            \n- The max rectangle area corresponds to a percentage of 100%.",
    
    "Features distributions": "- Input Features Z-scores distributions averaged across replicates,\
                    are grouped by Histone Modifications and gene expression (RNA).\
                        \n- Y-axis limits are set to the 0.1 and 99.9 percentiles of the data.",
    
    "MetaGene plots": "- TSS meta-plots display each Histone Modification (HM) across distinct Cell Types (CT),\
            with dashed lines representing\
            the corresponding control (Whole Cell Extract) signals for each CT. \
                \n- The y-axis maximum is consistent\
            among clusters, set to the maximum y-value among all clusters for each HM and CT combination.\
                \n-  averaged ChIP-seq metaplots centered on the Transcription Start Site (TSS)\
                were generated independently for each HM (and control) across all CTs, using SeqCode\
                (produceTSSplots). A single replicate BAM file was used for each HM/CT combination. The\
                fragment length was set to 250 bp, with a sliding window of 50 bp, covering a region from -\
                2500 to +2500 bp relative to the TSS. Signal profiles were smoothed using a moving rolling\
                mean with a window size of 200 bp. To make the plots scales as much comparable as possible\
                the maximum y-axis value was set for each CT-HM combination as the maximum value among\
                all clusters.",
                
    
    "Cluster categories": "- On the TOP: Tree maps depicting the percentage of genes in each cluster defined as most variable\
        and most stable in term of expression coefficient of variation . Five lists were generated: the most variable genes (top 4,000 CV) stratified by their\
        CTmax values (ESC, MES, CP, CM) and the most stable genes (bottom 4,000 CV).\
            \n- On the BOTTOM: Tree maps depicting the percentage of genes in each cluster annotated with an active,\
            bivalent or other chromatin state (in ESC) in another study (Gonzalez, 2021).\
            \n- The max rectangle area corresponds to a percentage of 100%.",
    
    "Expression of selected genes": "- Expression patterns of 18 randomly selected genes from the cluster (or custom selection). Each dot represents a replicate.",
    
    "Functional Term Enrichment Analysis": "- Term enrichment analyses were conducted using GSEApy in Python (via the EnrichR API)\
                            \n- The background list included all genes from the dataset (n=14996). \
                            \n- Only the top 5 significant terms were reported for each gene set, based on the adjusted p-value (adj. p < 0.05).",
    
    "Information of the Cluster": "Information of the Cluster"
}

# Define session state variables for `k` and `gene_query`
if "k" not in st.session_state:
    st.session_state.k = 76  # Default cluster
if "gene_query" not in st.session_state:
    st.session_state.gene_query = ""

# Function to update `k` from gene query and clear `gene_query`
def update_cluster_and_clear_query(k_value):
    st.session_state.k = k_value  # Update `k` in session state
    st.session_state.gene_query = ""  # Clear the query after updating


    

if 'expand_states' not in st.session_state:
    st.session_state.expand_states = {'bool':False, 'icon': ":material/expand:", 'label': 'Expand all'}  # Default to collapsed


# Callback function for the expand button
def expand_callback():
    st.session_state.expand_states['bool'] = not st.session_state.expand_states['bool']
    if st.session_state.expand_states['bool']:
        st.session_state.expand_states['icon'] = ":material/compress:"
        st.session_state.expand_states['label'] = 'Collapse all'
    else:
        st.session_state.expand_states['icon'] = ":material/expand:" 
        st.session_state.expand_states['label'] = 'Expand all'
    

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

    st.button(st.session_state.expand_states['label'], icon=st.session_state.expand_states['icon'], on_click=expand_callback, key="expand_button")



# --------------------------------------------------------------

st.markdown("<h1 style='text-align: center;'>Cluster Navigator</h1>", unsafe_allow_html=True,)
st.markdown("<h5 style='text-align: center;'>Explore genes in each cluster in term of input features distributions, Metagene plots...</h5>", unsafe_allow_html=True,)

#-------------------Clusters composition-------------------#
with st.expander("Cluster categories Map",  icon=":material/stacked_bar_chart:",expanded=st.session_state.expand_states['bool']):

    title_with_help('Genes distribution among clusters by categories', HELP_DICT['Cluster categories Map'])

    C = st.columns(2)

    # Cluster composition file viewers
    CV_file = f"./data/plots/Clusters_CV.pdf"
    Gonzalez_file = f"./data/plots/Clusters_Gonzalez.pdf"

    with C[0]:

        
        image = convert_pdf_to_image(CV_file)
        if image:
            st.image(image, use_container_width=True)
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
        image = convert_pdf_to_image(Gonzalez_file)
        if image:
            st.image(image, use_container_width=True)
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


st.divider()

st.markdown(f"<h5 style='text-align: center;'>Select a cluster from the sidebar</h5>", unsafe_allow_html=True)   
st.markdown(f"<h3 style='text-align: center;'>Cluster {k} (n= {NUM_OF_GENES})</h3>", unsafe_allow_html=True)   

#------------------------------------------Features distributions------------------------------------------#

#------------------------------------------Features distributions------------------------------------------#
with st.expander("Features distributions",  icon=":material/bar_chart:",expanded=st.session_state.expand_states['bool']):
    title_with_help('Features distributions', HELP_DICT['Features distributions'])

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

with st.expander("MetaGene plots", icon=":material/area_chart:", expanded=st.session_state.expand_states['bool']):
    title_with_help('MetaGene Plots', HELP_DICT['MetaGene plots'])

    tss_plot_pdf_file = f"./data/plots/TSSplots/C{k}_ext.pdf"

    tss_plot = convert_pdf_to_image(tss_plot_pdf_file)

    if tss_plot:
        st.image(tss_plot, use_container_width=True)

    # Add a download button for the PDF
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



#------------------------------------------TSS and Categories------------------------------------------#
with st.expander("Cluster categories",  icon=":material/stacked_bar_chart:", expanded=st.session_state.expand_states['bool']):
        title_with_help('Categories proportions', HELP_DICT['Cluster categories'])

        bar_comp= plot_stacked_bar(DATA[DATA['Cluster'] == k], ["ESC_ChromState_Gonzalez2021","CV_Category"] , COLOR_DICTS)

        st.plotly_chart(bar_comp, use_container_width=True)

        plotly_download_pdf(bar_comp, file_name=f"C{k}_CategoriesStackedBar.pdf")

#--------------------------------------------------------------
with st.expander("Expression of selected genes", icon=":material/timeline:", expanded=st.session_state.expand_states['bool']):
    title_with_help('Expression of selected genes', HELP_DICT['Expression of selected genes'])

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
with st.expander("Functional Term Enrichment Analysis", icon=":material/hdr_strong:", expanded=st.session_state.expand_states['bool']):
    title_with_help('Functional Term Enrichment Analysis', HELP_DICT['Functional Term Enrichment Analysis'])

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
with st.expander("Information of the Cluster", icon=":material/table_rows:",expanded=st.session_state.expand_states['bool']):
    title_with_help('Information of the Cluster', f"[Data tables description]({HOME_LINK}/Data#data-tables-description)")
    df_tabs(SEL.drop(columns='Cluster'))
    





    
    

add_footer()