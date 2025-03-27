from utils.my_module import *
st.markdown("<h2 style='text-align: center;'>Glossary</h3>", unsafe_allow_html=True)

Glossary = """
#### **Experiments/Histone Modifications (HM):**
- **RNA**: Gene expression levels  
- **H3K4me3**: Histone H3 with trimethylation at lysine 4  
- **H3K27ac**: Histone H3 with acetylation at lysine 27  
- **H3K27me3**: Histone H3 with trimethylation at lysine 27  

#### **Cell Types:**
- **ESC**: Embryonic Stem Cells  
- **MES**: Mesoderm  
- **CP**: Cardiac Progenitors  
- **CM**: Cardiomyocytes  

#### **Other Abbreviations:**
- **LogFC**: Log Fold Change  
- **FPKM**: Fragments Per Kilobase of transcript per Million mapped reads  
- **VAE**: Variational Autoencoder  
- **UMAP**: Uniform Manifold Approximation and Projection  
- **PCA**: Principal Component Analysis  
- **CV**: Coefficient of Variation  
- **RMSE**: Root Mean Squared Error  """
st.markdown(Glossary, unsafe_allow_html=True)

#-------------------Data tables description-------------------#
st.markdown("<h2 style='text-align: center;'>Data description</h3>", unsafe_allow_html=True)

DESCRIPTION= """
- #### **Input Features Z-scores**  
   Displays Z-scores for RNA expression and histone modifications (`H3K4me3`, `H3K27ac`, `H3K27me3`) across cell types. For simplicity the average value across replicates is shown.
   - **Format**: `{Experiment}_{CellType}` (e.g., `RNA_ESC`, `H3K4me3_MES`).  

- #### **RNA LogFCs**  
   Contains log fold change values comparing RNA expression between cell types. LogFC measures relative expression differences.  
   - **Format**: `RNA_{CellType1}_{CellType2}_FC` (e.g., `RNA_CM_CP_FC`: Log fold change of gene expression between cardiomyocites and cardiac progenitor cells.)

- #### **Gene Annotations**  
     - `Cluster`: Gene cluster assignment.  
     - `RNA_CV`: Coefficient of variation for RNA expression.  
     - `CV_Category`: Category based on RNA_CV.  
        - STABLE: Genes within the lower 4,000 of RNA_CV values.
        - ESC, MES, CP, CM: Genes in the top 4,000, assigned based on the cell type where they show maximum expression (e.g., ESC for genes peaking in embryonic stem cells).
        - Other: Genes with intermediate RNA_CV values, not classified into the above categories.
     - `ESC_ChromState_Gonzalez2021`: Chromatin state in embryonic stem cells based on the Gonzalez 2021 dataset.  
        - Active: Regions associated with open chromatin and active transcription.
        - Bivalent: Regions marked by both active and repressive histone modifications, indicative of poised regulatory elements.
        - Other: Regions not classified as Active or Bivalent, potentially representing repressed or uncharacterized chromatin states.
     - `VAE_RMSE`: Reconstruction error from the VAE model.  
     - `VAE_Sc`: Reconstruction cosine similarity derived from the VAE model.

- #### **VAE Latent Space**  
   Represents compressed gene features from Variational Autoencoder (VAE). Includes latent variables (`VAE1`-`VAE6`), UMAP projections (`VAE_UMAP1`, `VAE_UMAP2`), and PCA components (`VAE_PCA1`, `VAE_PCA2`).  

- #### **RNA-seq FPKMs**  
   Contains FPKM values for RNA expression across cell types and replicates. FPKM normalizes expression by transcript length and sequencing depth.  
   - **Format**: `RNA_{CellType}_{Replicate}` (e.g., `RNA_ESC_1`).   """
   
st.markdown(DESCRIPTION, unsafe_allow_html=True)

#---------------------------------#
st.divider()
st.markdown(f"<h3 style='text-align: center;'>VAE model architecture</h3>", unsafe_allow_html=True)   

fig1 = convert_pdf_to_image('./data/plots/fig1.pdf',)
if fig1:
    st.image(fig1, use_container_width=True)

#---------------------------------#
st.divider()
st.markdown(f"<h3 style='text-align: center;'>VAE model architecture</h3>", unsafe_allow_html=True)   

fig2 = convert_pdf_to_image('./data/plots/fig2.pdf',)
if fig2:
    st.image(fig2, use_container_width=True)

#---------------------------------#
st.divider()
st.markdown(f"<h3 style='text-align: center;'>VAE model architecture</h3>", unsafe_allow_html=True)   

fig3 = convert_pdf_to_image('./data/plots/fig3.pdf',)
if fig3:
    st.image(fig3, use_container_width=True)
    
    
    
add_footer()