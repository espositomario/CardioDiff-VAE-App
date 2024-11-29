from utils.my_module import *

st.set_page_config(layout="wide", initial_sidebar_state='collapsed')



C = st.columns(2)
CV_file = f"./data/plots/Clusters_CV.pdf"
Gonzalez_file = f"./data/plots/Clusters_Gonzalez.pdf"

with C[0]:
    st.markdown("<h3 style='text-align: center;'>CV</h3>", unsafe_allow_html=True,
                help="...")
    pdf_viewer(CV_file)
    try:
        with open(CV_file, "rb") as pdf_file:
            CV_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=CV_data,
            file_name=f"CV_Categories_Clusters_Intersection.pdf",
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.error("file not found.")

with C[1]:
    st.markdown("<h3 style='text-align: center;'>Gonzalez</h3>", unsafe_allow_html=True,
                help="...")
    pdf_viewer(Gonzalez_file)
    try:
        with open(Gonzalez_file, "rb") as pdf_file:
            Gonzalez_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=Gonzalez_data,
            file_name=f"Gonzalez_Categories_Clusters_Intersection.pdf",
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.error("file not found.")
        
        
        

#--------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Cluster visualization</h1>", unsafe_allow_html=True)

# Create two columns for layout
CC = st.columns(5)
    
with CC[1]:
    # Input field for selecting k (from 0 to 79)
    k = st.number_input(label="Select a Cluster (from 0 to 79)", label_visibility="collapsed"
                        , min_value=0, max_value=79, step=1, value=76, placeholder="Enter a number between 0 and 79")

    NUM_OF_GENES = GENE_CLUSTERS[str(k)]['len']
    GENE_LIST = GENE_CLUSTERS[str(k)]['gene_list']

with CC[2]:
    # Convert the list to a string with each element on a new line
    file_content = "\n".join(GENE_LIST)

    st.download_button(
        label="Gene List",
        icon=":material/download:",
        data=file_content,
        file_name=f"C{k}_GeneIDsList.txt",
        mime="text/plain"
    )

    
with CC[4]:
    # Add a tool to query by GENE
    with st.popover("🔍 Cluster by Gene"):
        st.markdown("<h3 style='text-align: center;'>Find where is a Gene</h3>", unsafe_allow_html=True)
        gene_query = st.text_input("Enter a mouse Refseq Gene symbol (e.g., Pim1)")

        if gene_query:
            if gene_query in DATA.index:
                k = DATA.loc[gene_query, 'Cluster']
                st.write(f"{gene_query} is in Cluster {k}.")
            else:
                st.write(f"Gene symbol: '{gene_query}' not found in the data. (Only mouse RefSeq gene symbols are accepted, notice that not all genes were included in the dataset)")

#--------------------------------------------------------------            
st.markdown(f"<h3 style='text-align: center;'>Cluster {k} (n= {NUM_OF_GENES})</h3>", unsafe_allow_html=True)   
    
C = st.columns(2)
with C[0]:    
    
    # Define file paths
    tss_plot_pdf_file = f"./data/plots/TSSplots/C{k}_ext.pdf"
    ora_plot_pdf = f"./data/plots/ORA/Cluster_{k}.pdf"


    st.markdown("<h3 style='text-align: center;'>TSS Plot</h3>", unsafe_allow_html=True,
                help="Transcription Starting Site Metaplots.\n - target histone marks on the rows\n - cell types in columns\n - dashed lines represents the control")
    pdf_viewer(tss_plot_pdf_file)
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


with C[1]:

    st.markdown("<h3 style='text-align: center;'>Term Enrichment</h3>", unsafe_allow_html=True)
    if os.path.exists(ora_plot_pdf):
        pdf_viewer(ora_plot_pdf)
        with open(ora_plot_pdf, "rb") as pdf_file:
            ora_data = pdf_file.read()
        st.download_button(
            label="",
            icon=":material/download:",
            data=ora_data,
            file_name=f"C{k}_TermEnrichment.pdf",
            mime="application/pdf",
        )
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
with st.expander("Cluster Data Table"):
    SEL = DATA[DATA['Cluster'] == k]
    df_tabs(SEL)



