from utils.my_module import *

st.markdown("<h1 style='text-align: center;'>Ontologies in the VAE Latent Space</h1>", unsafe_allow_html=True)


C = st.columns(2, gap='medium', vertical_alignment='top')


# Load the list of available mouse databases
DB_LIST = gp.get_library_name(organism='Mouse')
with C[0]:
    SEL_DB = st.selectbox("Database:", DB_LIST,index=DB_LIST.index('KEGG_2019_Mouse'), help='Enrichr API by gseapy python library')

# Once a database is selected, load the gene set
if SEL_DB:
    with st.spinner(f"Loading gene sets from **{SEL_DB}**..."):
        DB = gp.get_library(name=SEL_DB, organism='Mouse')
    
    GENE_SETS = list(DB.keys())
    with C[1]:
        # Step 3: Multiselect for gene sets
        SEL_GENE_SET = st.selectbox("Gene Set:", GENE_SETS)

    # Step 4: Display the genes from the selected gene sets
    if SEL_GENE_SET:
        GENE_LIST = hum2mouse(DB[SEL_GENE_SET])
        st.write(f"{GENE_LIST}")


    # Assuming you have DATA.index representing genes in your dataset
    TOTAL_GENES_IN_SET = len(GENE_LIST)
    GENE_LIST = set(DATA.index) & set(GENE_LIST)
    INTERSECT_COUNT = len(GENE_LIST)
    
    with C[1]:
        CC= st.columns([2,1], vertical_alignment='center')
        with CC[0]:
            st.write(f"Num. of genes: **{TOTAL_GENES_IN_SET}** (Intersecting with dataset: **{INTERSECT_COUNT}/{TOTAL_GENES_IN_SET}**)")
        with CC[1]:
            download_genes_list(GENE_LIST, key='GeneSet Download', filename=f'GeneIDs_{SEL_DB}_{SEL_GENE_SET}.txt')














add_footer()
