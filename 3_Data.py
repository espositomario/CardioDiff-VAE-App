from utils.my_module import *


def download_table(DATA):
    """
    Creates a button to download the provided DataFrame as a CSV file.

    Parameters:
    - DATA: pandas DataFrame to be downloaded.

    Returns:
    - None
    """
    # Convert the DataFrame to a CSV string
    csv_data = DATA.to_csv(index=True, index_label="GeneSymbol")

    # Create the download button
    st.download_button(
        label=" DataTable CSV",
        icon=":material/download:",  
        data=csv_data,
        file_name="CardioDiffVAE_data.csv",
        mime="text/csv",  # Correct MIME type for CSV files
        key='download_table'
    )


with st.sidebar:
    download_table(DATA)



#-------------------Filter data by genes (rows)-------------------#
st.markdown("<h3 style='text-align: center;'>Filter data by genes (rows)</h3>", unsafe_allow_html=True)

with st.expander("Select or Upload a gene list"):
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

df_tabs(DATA)



