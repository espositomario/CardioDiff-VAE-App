import streamlit as st
import plotly.express as px
import pandas as pd
import pickle
import os
from plotly.graph_objects import Figure, Violin
import plotly.graph_objects as go
from streamlit_pdf_viewer import pdf_viewer

# Load gene cluster dictionary
with open(f'./gene_clusters_dict.pkl', 'rb') as f:
    GENE_CLUSTERS = pickle.load(f)

# Load CODE and LOG matrices
CODE = pd.read_csv(f'./CODE.csv', index_col='GENE')

# Map cluster IDs to CODE and LOG
gene_to_cluster = {}
for cluster_id, gene_list in GENE_CLUSTERS.items():
    for gene in gene_list['gene_list']:
        gene_to_cluster[gene] = cluster_id

CODE["GMM_VAE_80"] = CODE.index.map(gene_to_cluster).astype(int)

# List of continuous features for coloring
continuous_features = ["RNA_CV", "VAE_RMSE", "VAE_Sc"]

# Streamlit app layout
st.set_page_config(layout="wide")

# Add a sidebar to select the page
page = st.sidebar.radio("Select Page", ["UMAP & Distribution", "Cluster TSS & ORA"])

if page == "UMAP & Distribution":
    st.title("Gene Visualization in VAE 6D Latent Space")

    # Dropdown for feature selection
    selected_feature = st.selectbox(
        "Select Continuous Feature to Color By",
        options=continuous_features,
        index=0  # Default to the first feature
    )

    # Filter data using color scale
    color_min, color_max = st.slider(
        f"Filter genes (points) by {selected_feature}",
        min_value=float(CODE[selected_feature].min()),
        max_value=float(CODE[selected_feature].max()),
        value=(float(CODE[selected_feature].min()), float(CODE[selected_feature].max())),
        step=0.01
    )
    filtered_data = CODE[(CODE[selected_feature] >= color_min) & (CODE[selected_feature] <= color_max)]

    # Calculate the number of selected genes
    num_selected_genes = filtered_data.shape[0]
    total_genes = CODE.shape[0]
    scatter_title = f"({num_selected_genes}/{total_genes} genes selected)"

    # Layout for scatter plot and violin plot
    col1, col2 = st.columns(2)

    # UMAP Scatter plot settings
    with col1:
        st.markdown("<h3 style='text-align: center;'>UMAP 2D Projection</h3>", unsafe_allow_html=True)
        with st.expander("⚙️", expanded=False):
            point_size = st.slider("Point Size", min_value=1, max_value=6, value=3, step=1)
            colormap = st.selectbox("Select Colormap", ["Spectral_r", "viridis", "plasma", "cividis", "rainbow", "magma"], index=0)

        # Create scatter plot
        fig = px.scatter(
            filtered_data,
            x="UMAP1",
            y="UMAP2",
            color=selected_feature,
            hover_data=["UMAP1", "UMAP2", selected_feature],
            title=scatter_title,
            labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
            color_continuous_scale=colormap
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            xaxis_showgrid=False, yaxis_showgrid=False,
            xaxis_tickvals=[], yaxis_tickvals=[],
            plot_bgcolor="white", autosize=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # Violin plot settings
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{selected_feature} Distribution for All Genes</h3>", unsafe_allow_html=True)
        box_violin_fig = go.Figure()
        box_violin_fig.add_trace(
            go.Violin(
                y=CODE[selected_feature],
                box=dict(visible=True),
                meanline=dict(visible=True),
                line_color="gray",
                fillcolor="lightgray",
                opacity=1
            )
        )
        box_violin_fig.add_shape(
            type="rect", xref="paper", yref="y", x0=0, x1=1, y0=color_min, y1=color_max,
            fillcolor="blue", opacity=0.10, line=dict(width=0)
        )
        box_violin_fig.update_layout(
            yaxis_title=selected_feature,
            xaxis=dict(showticklabels=False),
            plot_bgcolor="white", showlegend=False,
            autosize=True
        )
        st.plotly_chart(box_violin_fig, use_container_width=False)

elif page == "Cluster TSS & ORA":
    st.title("Cluster TSS and Term Enrichment")

    # Input field for selecting k (from 0 to 79)
    k = st.number_input("Enter a number (k) between 0 and 79:", min_value=0, max_value=79, step=1, value=0)
    

    # Define file paths
    tss_plot_pdf_file = f"./data/plots/TSSplots/C{k}_ext.pdf"
    ora_plot_pdf = f"./data/plots/ORA/Cluster_{k}.pdf"

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Left column: TSS plot
    with col1:
        st.markdown("<h3 style='text-align: center;'>TSS Plot</h3>", unsafe_allow_html=True)
        pdf_viewer(tss_plot_pdf_file)
        try:
            with open(tss_plot_pdf_file, "rb") as pdf_file:
                tss_data = pdf_file.read()
            st.download_button(
                label="Download",
                data=tss_data,
                file_name=f"C{k}_TSSPlot.pdf",
                mime="application/pdf",
            )
        except FileNotFoundError:
            st.error("TSS plot file not found.")

    # Right column: ORA plot
    with col2:
        st.markdown("<h3 style='text-align: center;'>Term Enrichment</h3>", unsafe_allow_html=True)
        if os.path.exists(ora_plot_pdf):
            pdf_viewer(ora_plot_pdf)
            with open(ora_plot_pdf, "rb") as pdf_file:
                ora_data = pdf_file.read()
            st.download_button(
                label="Download",
                data=ora_data,
                file_name=f"C{k}_TermEnrichment.pdf",
                mime="application/pdf",
            )
        else:
            st.write("No significant term resulted.")
