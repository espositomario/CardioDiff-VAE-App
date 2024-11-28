from home import *

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
    min_value=float(DATA[selected_feature].min()),
    max_value=float(DATA[selected_feature].max()),
    value=(float(DATA[selected_feature].min()), float(DATA[selected_feature].max())),
    step=0.01
)
filtered_data = DATA[(DATA[selected_feature] >= color_min) & (DATA[selected_feature] <= color_max)]

# Calculate the number of selected genes
num_selected_genes = filtered_data.shape[0]
total_genes = DATA.shape[0]
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
        x="VAE_UMAP1",
        y="VAE_UMAP2",
        color=selected_feature,
        hover_data=["VAE_UMAP1", "VAE_UMAP2", selected_feature],
        title=scatter_title,
        labels={"VAE_UMAP1": "UMAP1" , "VAE_UMAP2": "UMAP2"},
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
            y=DATA[selected_feature],
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