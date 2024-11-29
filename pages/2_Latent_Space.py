from utils.my_module import *
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

st.title("Gene encoding in VAE 6D Latent Space")

C = st.columns(2)

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