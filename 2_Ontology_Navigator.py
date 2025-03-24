from utils.my_module import *

st.markdown("<h1 style='text-align: center;'>Ontologies in the VAE Latent Space</h1>", unsafe_allow_html=True)


C = st.columns(2, gap='medium', vertical_alignment='top')


DB_LIST = sorted(gp.get_library_name(organism='Mouse'), key=str.lower)
with C[0]:
    SEL_DB = st.selectbox("Database:", DB_LIST,index=DB_LIST.index('WikiPathways_2024_Mouse'), 
                            help='Enrichr API by gseapy python library',
                            placeholder="Select or Type a database")

if SEL_DB:
    with st.spinner(f"Loading gene sets from **{SEL_DB}**..."):
        DB = gp.get_library(name=SEL_DB, organism='Mouse')
    
    GENE_SETS = sorted(list(DB.keys()), key=str.lower)
    with C[1]:
        # Step 3: Multiselect for gene sets
        SEL_GENE_SET = st.selectbox("Gene Set:", GENE_SETS, 
                                    index= GENE_SETS.index('Heart Development WP2067') if SEL_DB == 'WikiPathways_2024_Mouse'  else None,
                                    placeholder="Select or Type a gene set")


    if SEL_GENE_SET:
        GENE_LIST = hum2mouse(DB[SEL_GENE_SET])

        
        TOTAL_GENES_IN_SET = len(GENE_LIST)
        GENE_LIST = set(DATA.index) & set(GENE_LIST)
        INTERSECT_COUNT = len(GENE_LIST)
    
        with C[1]:
            CC= st.columns([2,1], vertical_alignment='center')
            with CC[0]:
                st.write(f"Num. of genes: **{TOTAL_GENES_IN_SET}** (Intersecting with dataset: **{INTERSECT_COUNT}/{TOTAL_GENES_IN_SET}**)")
            with CC[1]:
                download_genes_list(GENE_LIST, key='GeneSet Download', filename=f'GeneIDs_{SEL_DB}_{SEL_GENE_SET}.txt')


        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{SEL_GENE_SET} genes highligthed in the VAE latent space projection</h3>", unsafe_allow_html=True)   

        
        #
        C = st.columns([3,1], vertical_alignment='bottom', gap='medium')
        with C[0]:
            DR = st.selectbox("Select dimensionality reduction method", ["UMAP", "PCA"])
        with C[1]:
            with st.popover("Highlighted points/genes", icon=":material/settings:",):

                SHOW_LABELS = st.checkbox("Show gene labels", value=False)

                SEL_GENES_SIZE = st.slider("Point size", min_value=6, max_value=24, value=10, step=2)
                
                SEL_POINT_ALPHA = st.slider("Point transparency", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

                if SHOW_LABELS: LABEL_SIZE = st.slider("Label size", min_value=8, max_value=20, value=10, step=2)
                else: LABEL_SIZE = None



        C = st.columns(2,gap="large")

        with C[0]:    
            KEY1 = 'key1'
            
            fig1 = scatter(DATA, COLOR_FEATURES, SEL_GENES=GENE_LIST, DR=DR, key=KEY1+'popover', COLOR_DICTS=COLOR_DICTS, default_index=23, 
                            LABELS= SHOW_LABELS, SEL_GENES_SIZE=SEL_GENES_SIZE, LABEL_SIZE=LABEL_SIZE, DEF_POINT_ALPHA=0.3, SEL_POINT_ALPHA=SEL_POINT_ALPHA)
            st.plotly_chart(fig1, use_container_width=True,key=KEY1+'fig')



        with C[1]:   
            KEY2 = 'key2'
            # Dropdown for feature selection

            fig2 = scatter(DATA, COLOR_FEATURES, SEL_GENES=GENE_LIST, DR=DR, key = KEY2+'popover',COLOR_DICTS=COLOR_DICTS, default_index=24,
                            LABELS= SHOW_LABELS, SEL_GENES_SIZE=SEL_GENES_SIZE, LABEL_SIZE=LABEL_SIZE, DEF_POINT_ALPHA=0.3, SEL_POINT_ALPHA=SEL_POINT_ALPHA)
            st.plotly_chart(fig2, use_container_width=True,key=KEY2+'fig')
            

    





add_footer()
