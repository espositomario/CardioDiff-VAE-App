from utils.my_module import *
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center;'>Gene encoded in the VAE Latent Space</h1>", unsafe_allow_html=True)

C = st.columns(2,gap="large")

with C[0]:    
    KEY1 = 'key1'
    # Dropdown for feature selection
    SEL_FEAT_1 = st.selectbox(
        "Color By", key=KEY1+'sel_box',
        options=COLOR_FEATURES,
        index=3  # Default to the first feature
    )
    fig1 = scatter(DATA, SEL_FEAT_1,key=KEY1+'popover', COLOR_DICTS=COLOR_DICTS)
    st.plotly_chart(fig1, use_container_width=True,key=KEY1+'fig')

with C[1]:    
    
    

    KEY2 = 'key2'
    # Dropdown for feature selection
    SEL_FEAT_2 = st.selectbox(
        "Color By", key=KEY2+'sel_box',
        options=COLOR_FEATURES,
        index=2  # Default to the first feature
    )
    fig2 = scatter(DATA, SEL_FEAT_2,KEY2+'popover',COLOR_DICTS=COLOR_DICTS)
    st.plotly_chart(fig2, use_container_width=True,key=KEY2+'fig')
    
    
    
SEL_GENES= st.multiselect("Select a gene", options=DATA.index, 
                max_selections=12,
                key="select a gene")
st.write(", ".join(SEL_GENES))