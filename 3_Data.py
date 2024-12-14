from utils.my_module import *




with st.sidebar:
    download_table(DATA)



#-------------------Filter data by genes (rows)-------------------#
st.markdown("<h3 style='text-align: center;'>Filter data by genes (rows)</h3>", unsafe_allow_html=True)

#with st.expander("Select or Upload a gene list"):
SEL_GENES = select_genes()
# Plot Sankey diagram 
if SEL_GENES:
    # filter DATA by sekected genes
    st.dataframe(DATA.loc[SEL_GENES])
    # Plot Sankey diagram
    st.markdown("<h5 style='text-align: center;'>Gene to Cluster Sankey Diagram</h5>", unsafe_allow_html=True)
    fig = plot_sankey(DATA, SEL_GENES, font_color="white", font_size=14, link_opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


#-------------------Filter data by features (columns)-------------------#
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Filter data by features (columns)</h3>", unsafe_allow_html=True)
#with st.expander("Select Features to display"):
df_tabs(DATA)


#-------------------Clusters composition-------------------#
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Genes distribution among clusters by categories</h3>", unsafe_allow_html=True)

#with st.expander("Clusters categories Maps", expanded=True):

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
