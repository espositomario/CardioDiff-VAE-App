from utils.my_module import *

st.title("Home")





col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <a href="{HOME_LINK}/Latent_Space" style="text-decoration: none;">
            <img src="./data/plots/page1.jpg" width="100%" onclick="window.location.href='{HOME_LINK}/Latent_Space'">
        </a>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <a href="{HOME_LINK}/Ontology_Navigator" style="text-decoration: none;">
            <img src="./data/plots/page3.jpg" width="100%" onclick="window.location.href='{HOME_LINK}/Ontology_Navigator'">
        </a>
        """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <a href="{HOME_LINK}/Clusters_Navigator" style="text-decoration: none;">
            <img src="./data/plots/page2.jpg" width="100%" onclick="window.location.href='{HOME_LINK}/Clusters_Navigator'">
        </a>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <a href="{HOME_LINK}/Data" style="text-decoration: none;">
            <img src="./data/plots/page4.jpg" width="100%" onclick="window.location.href='{HOME_LINK}/Data'">
        </a>
        """, unsafe_allow_html=True)




#---------------------------------#
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>VAE model architecture</h3>", unsafe_allow_html=True)   

fig1 = convert_pdf_to_image('./data/plots/fig1.pdf',)
if fig1:
    st.image(fig1, use_container_width=True)

#---------------------------------#
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>VAE model architecture</h3>", unsafe_allow_html=True)   

fig2 = convert_pdf_to_image('./data/plots/fig2.pdf',)
if fig2:
    st.image(fig2, use_container_width=True)

#---------------------------------#
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>VAE model architecture</h3>", unsafe_allow_html=True)   

fig3 = convert_pdf_to_image('./data/plots/fig3.pdf',)
if fig3:
    st.image(fig3, use_container_width=True)
    
add_footer()