from utils.my_module import *

Glossary = """## Glossary

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