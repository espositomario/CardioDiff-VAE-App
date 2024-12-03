from utils.my_module import *
#st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center;'>Gene encoded in the VAE Latent Space</h1>", unsafe_allow_html=True)



SEL_GENES= st.multiselect("Select genes to highlight", options=DATA.index, 
                max_selections=20,
                key="select a gene")




def scatter(DATA, COLOR_FEATURES, SEL_GENES, key, COLOR_DICTS=None, default_index=0):
    import plotly.graph_objects as go

    # Layout for control elements
    C = st.columns([3, 1], vertical_alignment="bottom", gap="large")
    with C[0]:
        # Dropdown for feature selection
        selected_feature = st.selectbox(
            "Color By", key=key + 'sel_box',
            options=COLOR_FEATURES,
            index=default_index  # Default to the first feature
        )

    CAT = False
    if pd.api.types.is_categorical_dtype(DATA[selected_feature]) or DATA[selected_feature].dtype == 'object':
        CAT = True
        color_dict = COLOR_DICTS.get(selected_feature, None)

    with C[1]:
        with st.popover("", icon=":material/settings:"):
            point_size = st.slider("Point Size", min_value=1, max_value=8, value=3, step=1, key=key + 'point_size')
            point_opacity = st.slider("Transparency", min_value=0.1, max_value=1.0, value=0.9, step=0.1, key=key + 'point_opacity')
            if not CAT:
                colormap = st.segmented_control(
                    "Select Colormap", ["Spectral_r", "Blues", "viridis", "RdBu_r"],
                    default='Spectral_r', selection_mode='single', key=key + 'colormap'
                )
                min_col, max_col = st.slider(
                    "Color Range (percentile)",
                    min_value=1, max_value=99, value=(1, 99), step=1, key=key + 'min_max'
                )

    # Precompute colors for all points
    if CAT:
        # Categorical coloring
        colors = DATA[selected_feature].map(
            lambda val: color_dict[val] if color_dict and val in color_dict else "gray"
        ).tolist()
    else:
            
        # Compute min and max percentiles
        min_p, max_p = np.percentile(DATA[selected_feature], [min_col, max_col])
        # Clamp values between min_p and max_p, normalize, and map to the colormap
        feature_values = DATA[selected_feature].values
        clamped_values = np.clip(feature_values, min_p, max_p)
        normalized_values = (clamped_values - min_p) / (max_p - min_p)
        # Map normalized values to colors in one operation
        cmap = cm.get_cmap(colormap)
        # Map normalized values directly to RGBA using matplotlib
        rgba_colors = cmap(normalized_values)
        # Convert RGBA to HEX for compatibility with Plotly
        colors = [mcolors.to_hex(rgba) for rgba in rgba_colors]

    # Create the base scatter plot
    fig = px.scatter(
        DATA,
        x="VAE_UMAP1",
        y="VAE_UMAP2",
        color=selected_feature,
        hover_data=[DATA.index, "Cluster", selected_feature],
        title=f"{selected_feature}",
        labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
        color_discrete_map=color_dict if CAT and color_dict else None,
        color_continuous_scale=colormap if not CAT else None,
        range_color=[min_p, max_p] if not CAT else None,
    )
    fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))

    # Highlight selected genes
    if SEL_GENES:
        for gene in SEL_GENES:
            if gene in DATA.index:
                gene_data = DATA.loc[gene]
                gene_color = colors[DATA.index.get_loc(gene)]  # Retrieve precomputed color
                fig.add_trace(go.Scattergl(
                    x=[gene_data["VAE_UMAP1"]],
                    y=[gene_data["VAE_UMAP2"]],
                    mode="markers",
                    marker=dict(size=16, color=gene_color, line=dict(width=1, color="grey")),
                    showlegend=False,  # Avoid duplicate legend entries
                    hovertemplate=(
                        f"<b>Gene:</b> {gene}<br>"
                        f"<b>VAE_UMAP1:</b> {gene_data['VAE_UMAP1']}<br>"
                        f"<b>VAE_UMAP2:</b> {gene_data['VAE_UMAP2']}<br>"
                        f"<b>{selected_feature}:</b> {gene_data[selected_feature]}<br>"
                        "<extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor="red", font=dict(color="white"))  # Red hover background
                ))

                # Add annotation for the gene name with a white background
                fig.add_annotation(
                    x=gene_data["VAE_UMAP1"],
                    y=gene_data["VAE_UMAP2"],
                    text=gene,
                    #showarrow=True,
                    arrowsize=0.5,
                    arrowcolor="grey",
                    arrowhead=0,
                    font=dict(size=14, color="black"),  # Text font
                    bgcolor="white",  # Background color for the annotation
                    #bordercolor="black",  # Optional: Add border to the annotation
                    borderwidth=0,
                    borderpad=0,
                )


    # Customize layout
    fig.update_layout(
        title=None,
        margin=dict(t=0),
        xaxis=dict(showgrid=False, tickvals=[]),
        yaxis=dict(showgrid=False, tickvals=[]),
        plot_bgcolor="white",
        autosize=True
    )

    return fig











C = st.columns(2,gap="large")

with C[0]:    
    KEY1 = 'key1'
    
    fig1 = scatter(DATA, COLOR_FEATURES, SEL_GENES=SEL_GENES, key=KEY1+'popover', COLOR_DICTS=COLOR_DICTS, default_index=1)
    st.plotly_chart(fig1, use_container_width=True,key=KEY1+'fig')



with C[1]:   
    KEY2 = 'key2'
    # Dropdown for feature selection

    fig2 = scatter(DATA, COLOR_FEATURES, SEL_GENES=SEL_GENES, key = KEY2+'popover',COLOR_DICTS=COLOR_DICTS, default_index=2)
    st.plotly_chart(fig2, use_container_width=True,key=KEY2+'fig')
    
    
    

