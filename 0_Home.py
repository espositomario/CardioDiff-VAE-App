from utils.my_module import *

st.title("Home")


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