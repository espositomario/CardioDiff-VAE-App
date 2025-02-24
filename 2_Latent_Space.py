from utils.my_module import *

st.markdown("<h1 style='text-align: center;'>Genome in the VAE Latent Space</h1>", unsafe_allow_html=True)


DR = st.selectbox("Select Dimensionality Reduction", ["UMAP", "PCA"])



with st.expander("Highlight genes in the scatter plot", icon=":material/checklist:"):
    SHOW_LABELS  = False
    SEL_GENES_SIZE = 16
    LABEL_SIZE = 12
    C = st.columns([4,1])
    

    SEL_GENES = select_genes()

    with st.popover("", icon=":material/settings:"):
        if SEL_GENES: 
 
            SHOW_LABELS = st.checkbox("Show gene labels", value=True)

            SEL_GENES_SIZE = st.slider("Point size", min_value=8, max_value=24, value=16, step=2)

            if SHOW_LABELS: LABEL_SIZE = st.slider("Label size", min_value=8, max_value=20, value=12, step=2)









C = st.columns(2,gap="small")



with C[0]:    
    KEY1 = 'key1'
    
    fig1 = scatter(DATA, COLOR_FEATURES, SEL_GENES=SEL_GENES, DR=DR, key=KEY1+'popover', COLOR_DICTS=COLOR_DICTS, default_index=1, 
                    LABELS= SHOW_LABELS, SEL_GENES_SIZE=SEL_GENES_SIZE, LABEL_SIZE=LABEL_SIZE)
    st.plotly_chart(fig1, use_container_width=True,key=KEY1+'fig')



with C[1]:   
    KEY2 = 'key2'
    # Dropdown for feature selection

    fig2 = scatter(DATA, COLOR_FEATURES, SEL_GENES=SEL_GENES, DR=DR, key = KEY2+'popover',COLOR_DICTS=COLOR_DICTS, default_index=2,
                    LABELS= SHOW_LABELS, SEL_GENES_SIZE=SEL_GENES_SIZE, LABEL_SIZE=LABEL_SIZE)
    st.plotly_chart(fig2, use_container_width=True,key=KEY2+'fig')
    

    



add_footer()