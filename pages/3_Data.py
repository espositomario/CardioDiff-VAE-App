from home import * 



st.dataframe(filter_dataframe(DATA))

with st.expander('Info'):
    st.markdown("- VAE* columns are the 6D latent space coordinates of the genes.\n"
                 "- RNA_(CellType) columns are the RNA-seq FPKM values of the genes.\n"
                 "- ChIP-seq columns are the Z-scores of the ChIP-seq levels (average between replicates).\n"
                 "- Other columns are the annotations and other features of the genes.")

with st.expander('Features group data tables'):
    VAE_CODE = DATA.iloc[:, :6]

    Z_AVG = DATA.iloc[:, 6:22]

    # Create Streamlit tabs
    tabs = st.tabs(["All", "VAE 6D LatentSpace", "ChIP-levels"])

    with tabs[0]:
        st.markdown("Whole data table", help="This table contains all the input features, annotations, and other features for all genes.")
        st.dataframe(DATA)

    with tabs[1]:
        st.write("Gene encoding in VAE 6D Latent Space")
        st.dataframe(VAE_CODE)

    with tabs[2]:
        st.write("Z-score of ChIP-seq levels (Avg between replicates)")
        st.dataframe(Z_AVG)