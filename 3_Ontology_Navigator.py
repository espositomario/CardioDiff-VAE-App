from utils.my_module import *

st.markdown("<h1 style='text-align: center;'>Ontologies in the VAE Latent Space</h1>", unsafe_allow_html=True)

import gseapy as gp

# Load the list of available mouse databases
DB_LIST = gp.get_library_name(organism='Mouse')


# ropdown for selecting the database
SEL_DB = st.selectbox("Select a Mouse Database:", DB_LIST)

# Once a database is selected, load the gene set
if SEL_DB:
    st.write(f"Loading gene sets from **{SEL_DB}**...")
    DB = gp.get_library(name=SEL_DB, organism='Mouse')
    GENE_SETS = list(DB.keys())

    # Step 3: Multiselect for gene sets
    SEL_GENE_SET = st.selectbox("Select Gene Sets:", GENE_SETS)

    # Step 4: Display the genes from the selected gene sets
    if SEL_GENE_SET:
        st.write(f"Genes in the Selected Gene Set: {DB[SEL_GENE_SET]}")







add_footer()
