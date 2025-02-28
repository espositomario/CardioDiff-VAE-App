from utils.my_module import *




with st.sidebar:
    download_table(DATA)

    #get_gene_ncbi_page(NCBI_IDs)



#-------------------Filter data by genes (rows)-------------------#
st.markdown("<h2 style='text-align: center;'>Filter data by genes</h2>", unsafe_allow_html=True)

#with st.expander("Select or Upload a gene list"):
SEL_GENES = select_genes()
# Plot Sankey diagram 
if SEL_GENES:
    # filter DATA by sekected genes
    st.dataframe(DATA.loc[SEL_GENES], 
                        column_config={
                                    "NCBI": st.column_config.LinkColumn(
                                        "Gene Name",
                                        help="Click to open the NCBI Gene page",
                                        #validate=r"^https://[a-z]+\.streamlit\.app$",
                                        #max_chars=100,
                                        display_text=r"AND\+(.+?)%5BGene\+Name%5D"

                                        #display_text="NCBI-Gene"
                                    ),
                                },
                                hide_index=True,
    )
    #l
    
    
    
    st.divider()
    title_with_help('Features distributions', 'help_text')

    C = st.columns(4, gap="small")

    # Create a PdfPages object to store all the plots
    pdf_path = "/tmp/plots.pdf"
    pdf_pages = PdfPages(pdf_path)
    
    for i,feature in enumerate(['RNA', 'H3K4me3', 'H3K27ac', 'H3K27me3']):
        FILT_DF = DATA[[f"{feature}_{ct}" for ct in CT_LIST]].copy()
        VMIN, VMAX = FILT_DF.min().min(), FILT_DF.max().max()
        VMIN = FILT_DF.quantile([0.001]).min(axis=1)[0.001]
        VMAX = FILT_DF.quantile([0.999]).max(axis=1)[0.999]
        FILT_DF = FILT_DF.loc[SEL_GENES]
        
        with C[i]:
            fig, ax= plot_violin_box(FILT_DF, feature, CT_LIST, HM_COL_DICT, CT_COL_DICT, y_lab='Z-score' if i==0 else'',
                                        VMIN=VMIN, VMAX=VMAX)

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
            file_name=f"SelectedGenes_Feature_distributions.pdf",
            mime="application/pdf"
        )
    
    # Plot Sankey diagram
    st.markdown("<h3 style='text-align: center;'>Gene to Cluster Sankey Diagram</h3>", unsafe_allow_html=True)
    fig = plot_sankey(DATA, SEL_GENES, font_color="white", font_size=14, link_opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


#-------------------Filter data by features (columns)-------------------#
st.divider()
st.markdown("<h2 style='text-align: center;'>Filter data by features</h2>", unsafe_allow_html=True)
#with st.expander("Select Features to display"):
df_tabs(DATA)

st.markdown("<h2 style='text-align: center;'>Data tables description</h3>", unsafe_allow_html=True)

#-------------------Data tables description-------------------#
DESCRIPTION= """
- #### **Features Z-scores Table**  
   Displays Z-scores for RNA expression and histone modifications (`H3K4me3`, `H3K27ac`, `H3K27me3`) across cell types. For simplicity the average value across replicates is shown.
   - **Format**: `{Experiment}_{CellType}` (e.g., `RNA_ESC`, `H3K4me3_MES`).  

- #### **RNA LogFCs Table**  
   Contains log fold change values comparing RNA expression between cell types. LogFC measures relative expression differences.  
   - **Format**: `RNA_{CellType1}_{CellType2}_FC` (e.g., `RNA_CM_CP_FC`: Log fold change of gene expression between cardiac mesoderm and cardiac progenitor cells.)

- #### **Gene Annotations Table**  
     - `Cluster`: Gene cluster assignment.  
     - `RNA_CV`: Coefficient of variation for RNA expression.  
     - `CV_Category`: Category based on RNA variability.  
     - `ESC_ChromState_Gonzalez2021`: Chromatin state in embryonic stem cells based on the Gonzalez 2021 dataset.  
     - `VAE_RMSE`: Reconstruction error from the VAE model.  
     - `VAE_Sc`: A score derived from the VAE model.

- #### **VAE Latent Space Table**  
   Represents compressed gene features from Variational Autoencoder (VAE). Includes latent variables (`VAE1`-`VAE6`), UMAP projections (`VAE_UMAP1`, `VAE_UMAP2`), and PCA components (`VAE_PCA1`, `VAE_PCA2`).  

- #### **RNA-seq FPKMs Table**  
   Contains FPKM values for RNA expression across cell types and replicates. FPKM normalizes expression by transcript length and sequencing depth.  
   - **Format**: `RNA_{CellType}_{Replicate}` (e.g., `RNA_ESC_1`).   """
   
st.markdown(DESCRIPTION, unsafe_allow_html=True)

add_footer()