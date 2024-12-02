from utils.my_module import *
#st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center;'>Gene encoded in the VAE Latent Space</h1>", unsafe_allow_html=True)








def scatter(DATA, COLOR_FEATURES,key, COLOR_DICTS=None, default_index=0):
    
    C = st.columns([3,1], vertical_alignment="bottom", gap="large")
    with C[0]:
        # Dropdown for feature selection
        selected_feature = st.selectbox(
            "Color By", key=key+'sel_box',
            options=COLOR_FEATURES,
            index=default_index  # Default to the first feature
        )
    
    
    CAT = False
    if pd.api.types.is_categorical_dtype(DATA[selected_feature]) or DATA[selected_feature].dtype == 'object':
        CAT = True
        color_dict = COLOR_DICTS.get(selected_feature, None)
    with C[1]:
        with st.popover("", icon=":material/settings:"):
            point_size = st.slider("Point Size", min_value=1, max_value=8, value=3, step=1, key=key+'point_size')
            point_opacity = st.slider("Transparency", min_value=0.1, max_value=1.0, value=0.9, step=0.1, key=key+'point_opacity')
            if not CAT:
                colormap = st.segmented_control("Select Colormap", ["Spectral_r", "Blues","viridis", "RdBu_r"], 
                                                default= 'Spectral_r',selection_mode='single', key=key+'colormap')
                min_col, max_col = st.slider("Color Range (percentile)", 
                                                min_value=1, max_value=99, value=(1,99), step=1, key=key+'min_max')

    # Determine if the selected feature is categorical or continuous
    if CAT:
        # Handle categorical feature
        fig = px.scatter(
            DATA,
            x="VAE_UMAP1",
            y="VAE_UMAP2",
            color=selected_feature,
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
            color_discrete_map=color_dict if color_dict else None
        )
    else:
        # Handle continuous feature
        # Compute 1st and 99th percentiles for color scaling
        

        min_p, max_p = np.percentile(DATA[selected_feature], [min_col, max_col])
        fig = px.scatter(
            DATA,
            x="VAE_UMAP1",
            y="VAE_UMAP2",
            color=selected_feature,
            hover_data=[DATA.index, "Cluster", selected_feature],
            title=f"{selected_feature}",
            labels={"VAE_UMAP1": "UMAP1", "VAE_UMAP2": "UMAP2"},
            color_continuous_scale=colormap,
            range_color=(min_p, max_p),  # Apply percentile scaling
        )
    
    
    
    fig.update_traces(marker=dict(size=point_size, opacity=point_opacity))

    fig.update_layout(
        title=None,  # Remove the title
        margin=dict(t=0),  # Remove the top margin to eliminate space
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_tickvals=[], yaxis_tickvals=[],
        plot_bgcolor="white", autosize=True
    )
    return fig





    
    

C = st.columns(2,gap="large")

with C[0]:    
    KEY1 = 'key1'
    
    fig1 = scatter(DATA, COLOR_FEATURES,key=KEY1+'popover', COLOR_DICTS=COLOR_DICTS, default_index=1)
    st.plotly_chart(fig1, use_container_width=True,key=KEY1+'fig')



with C[1]:   
    KEY2 = 'key2'
    # Dropdown for feature selection

    fig2 = scatter(DATA, COLOR_FEATURES,KEY2+'popover',COLOR_DICTS=COLOR_DICTS, default_index=2)
    st.plotly_chart(fig2, use_container_width=True,key=KEY2+'fig')
    
    
    
SEL_GENES= st.multiselect("Select a gene", options=DATA.index, 
                max_selections=12,
                key="select a gene")
st.write(", ".join(SEL_GENES))