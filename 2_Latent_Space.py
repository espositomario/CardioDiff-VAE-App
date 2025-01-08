from utils.my_module import *

st.markdown("<h1 style='text-align: center;'>Genome in the VAE Latent Space</h1>", unsafe_allow_html=True)


DR = st.selectbox("Select Dimensionality Reduction", ["UMAP", "PCA"])



with st.expander("Highlight genes in the scatter plot", icon=":material/checklist:"):
    SHOW_LABELS  = False
    SEL_GENES_SIZE = 16
    LABEL_SIZE = 12
    C = st.columns([4,1])
    

    SEL_GENES = select_genes()

    with st.popover("", icon=":material/settings:"):
        if SEL_GENES: 
 
            SHOW_LABELS = st.checkbox("Show gene labels", value=True)

            SEL_GENES_SIZE = st.slider("Point size", min_value=8, max_value=24, value=16, step=2)

            if SHOW_LABELS: LABEL_SIZE = st.slider("Label size", min_value=8, max_value=20, value=12, step=2)







def scatter(DATA, COLOR_FEATURES, SEL_GENES, key, COLOR_DICTS=None, default_index=0, 
            LABELS=True, SEL_GENES_SIZE=16, LABEL_SIZE=12):
    

    # Layout for control elements
    C = st.columns([5,1, 1, 1], vertical_alignment="bottom", gap="small")
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

        if color_dict is None:
            categories = DATA[selected_feature].unique()
            num_categories = len(categories)

            hsv_colors = [colorsys.hsv_to_rgb(i / num_categories, 1, 1) for i in range(num_categories)]
            hex_colors = [mcolors.to_hex(rgb) for rgb in hsv_colors]
            color_dict = {cat: hex_colors[i] for i, cat in enumerate(categories)}
            
    if not CAT:
        with C[1]:
            with st.popover("", icon=":material/filter_list:"):
                # Range selection slider
                min_val = DATA[selected_feature].min()
                max_val = DATA[selected_feature].max()
                selected_range = st.slider(
                    f"Range Mask on {selected_feature}", min_value=min_val, max_value=max_val,
                    value=(min_val, max_val), key=key + 'range_slider'
                )

    
    
    with C[2]:
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
        # Apply range selection (masking)
        mask = (DATA[selected_feature] < selected_range[0]) | (DATA[selected_feature] > selected_range[1])
        colors = np.where(mask, "lightgrey", colors).tolist() #Mask the colors

        # Create the base scatter plot
    if CAT:
        fig = px.scatter(
            DATA,
            x="VAE_UMAP1",
            y="VAE_UMAP2",
            color=selected_feature,
            color_discrete_map=color_dict,
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
        )
        fig.update_traces(marker=dict(size=point_size, opacity=point_opacity,))
    else:
        fig = px.scatter(
            DATA,
            x="VAE_UMAP1",
            y="VAE_UMAP2",
            color=DATA[selected_feature],
            color_continuous_scale=colormap,
            range_color=[min_p, max_p],
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
        )
        mask = (DATA[selected_feature] < selected_range[0]) | (DATA[selected_feature] > selected_range[1])
        fig.update_traces(
            marker=dict(
                size=point_size, 
                opacity=np.where(mask, 0.1, point_opacity) #change opacity instead of color
            )
        )


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
                    marker=dict(size=SEL_GENES_SIZE, color=gene_color, line=dict(width=1, color="black")),
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

                if LABELS:
                    # Add annotation for the gene name with a white background
                    fig.add_annotation(
                        x=gene_data["VAE_UMAP1"],
                        y=gene_data["VAE_UMAP2"],
                        text=gene,
                        #showarrow=True,
                        arrowsize=0.5,
                        arrowcolor="black",
                        arrowhead=0,
                        font=dict(size=LABEL_SIZE, color="black"),  # Text font
                        bgcolor="white",  # Background color for the annotation
                        #bordercolor="black",  # Optional: Add border to the annotation
                        borderwidth=0,
                        borderpad=0,
                    
                    )


    # Customize layout
    fig.update_layout(
        title='',
        margin=dict(t=50, r=50, b=50, l=50),
        xaxis=dict(showgrid=False, zeroline=False,  tickvals=[]),
        yaxis=dict(showgrid=False, zeroline=False,   tickvals=[]),
        plot_bgcolor="white",
        autosize=False,
        showlegend=True #show legend only if CAT is true
    )

    with C[3]:
        plotly_download_png(fig, file_name=f"VAE_LatentSpace_UMAP_{selected_feature}.png")
    
    return fig





def scatter(DATA, COLOR_FEATURES, SEL_GENES, DR, key, COLOR_DICTS=None, default_index=0,
            LABELS=True, SEL_GENES_SIZE=16, LABEL_SIZE=12):
    
    COMPONENTS = [col for col in DATA.columns if f'{DR}' in col]
    # Layout for control elements
    C = st.columns([5, 1, 1, 1], vertical_alignment="bottom", gap="small")
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

        if color_dict is None:
            categories = DATA[selected_feature].unique()
            num_categories = len(categories)

            hsv_colors = [colorsys.hsv_to_rgb(i / num_categories, 1, 1) for i in range(num_categories)]
            hex_colors = [mcolors.to_hex(rgb) for rgb in hsv_colors]
            color_dict = {cat: hex_colors[i] for i, cat in enumerate(categories)}

    if not CAT:
        with C[1]:
            with st.popover("", icon=":material/filter_list:"):
                # Range selection slider
                min_val = DATA[selected_feature].min()
                max_val = DATA[selected_feature].max()
                selected_range = st.slider(
                    f"Range Mask on {selected_feature}", min_value=min_val, max_value=max_val,
                    value=(min_val, max_val), key=key + 'range_slider'
                )

    with C[2]:
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
        # Apply range selection (masking)
        mask = (DATA[selected_feature] < selected_range[0]) | (DATA[selected_feature] > selected_range[1])
        colors = np.where(mask, "lightgrey", colors).tolist() #Mask the colors

    if CAT:
        fig = px.scatter(
            DATA,
            x=COMPONENTS[0],  # Use dynamic component names
            y=COMPONENTS[1],
            color=selected_feature,
            color_discrete_map=color_dict,
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={COMPONENTS[0]: COMPONENTS[0].replace("VAE_", ""), COMPONENTS[1]: COMPONENTS[1].replace("VAE_", "")}, # Dynamic labels
        )
        fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))
    else:
        fig = px.scatter(
            DATA,
            x=COMPONENTS[0],  # Use dynamic component names
            y=COMPONENTS[1],
            color=DATA[selected_feature],
            color_continuous_scale=colormap,
            range_color=[min_p, max_p],
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={COMPONENTS[0]: COMPONENTS[0].replace("VAE_", ""), COMPONENTS[1]: COMPONENTS[1].replace("VAE_", "")}, # Dynamic labels
        )
        mask = (DATA[selected_feature] < selected_range[0]) | (DATA[selected_feature] > selected_range[1])
        fig.update_traces(
            marker=dict(
                size=point_size,
                opacity=np.where(mask, 0.1, point_opacity)
            )
        )

    # Highlight selected genes (modified)
    if SEL_GENES:
        for gene in SEL_GENES:
            if gene in DATA.index:
                gene_data = DATA.loc[gene]
                gene_color = colors[DATA.index.get_loc(gene)]
                fig.add_trace(go.Scattergl(
                    x=[gene_data[COMPONENTS[0]]],  # Use dynamic component names
                    y=[gene_data[COMPONENTS[1]]],
                    mode="markers",
                    marker=dict(size=SEL_GENES_SIZE, color=gene_color, line=dict(width=1, color="black")),
                    showlegend=False,  # Avoid duplicate legend entries
                    hovertemplate=(
                        f"<b>Gene:</b> {gene}<br>"
                        f"<b>{COMPONENTS[0]}:</b> {gene_data[COMPONENTS[0]]}<br>" # Dynamic hover data
                        f"<b>{COMPONENTS[1]}:</b> {gene_data[COMPONENTS[1]]}<br>"
                        f"<b>{selected_feature}:</b> {gene_data[selected_feature]}<br>"
                        "<extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor="red", font=dict(color="white"))  # Red hover background
                ))

                if LABELS:
                    fig.add_annotation(
                        x=gene_data[COMPONENTS[0]],
                        y=gene_data[COMPONENTS[1]],
                        text=gene,
                        #showarrow=True,
                        arrowsize=0.5,
                        arrowcolor="black",
                        arrowhead=0,
                        font=dict(size=LABEL_SIZE, color="black"),  # Text font
                        bgcolor="white",  # Background color for the annotation
                        #bordercolor="black",  # Optional: Add border to the annotation
                        borderwidth=0,
                        borderpad=0,
                    
                    )

    # Customize layout
    fig.update_layout(
        title='',
        margin=dict(t=50, r=50, b=50, l=50),
        xaxis=dict(showgrid=False, zeroline=False,  tickvals=[]),
        yaxis=dict(showgrid=False, zeroline=False,   tickvals=[]),
        plot_bgcolor="white",
        autosize=False,
        showlegend=True
    )

    with C[3]:
        plotly_download_png(fig, file_name=f"VAE_LatentSpace_{DR}_{selected_feature}")
    
    return fig


C = st.columns(2,gap="large")



with C[0]:    
    KEY1 = 'key1'
    
    fig1 = scatter(DATA, COLOR_FEATURES, SEL_GENES=SEL_GENES, DR=DR, key=KEY1+'popover', COLOR_DICTS=COLOR_DICTS, default_index=1, 
                    LABELS= SHOW_LABELS, SEL_GENES_SIZE=SEL_GENES_SIZE, LABEL_SIZE=LABEL_SIZE)
    st.plotly_chart(fig1, use_container_width=True,key=KEY1+'fig')



with C[1]:   
    KEY2 = 'key2'
    # Dropdown for feature selection

    fig2 = scatter(DATA, COLOR_FEATURES, SEL_GENES=SEL_GENES, DR=DR, key = KEY2+'popover',COLOR_DICTS=COLOR_DICTS, default_index=2,
                    LABELS= SHOW_LABELS, SEL_GENES_SIZE=SEL_GENES_SIZE, LABEL_SIZE=LABEL_SIZE)
    st.plotly_chart(fig2, use_container_width=True,key=KEY2+'fig')
    

    



add_footer()