from utils.my_module import *

st.divider()
st.markdown("<h2 style='text-align: center;'>TITLE</h3>", unsafe_allow_html=True)   
st.markdown("<h3 style='text-align: center;'>Mario Esposito, Luciano Di Croce and Enrique Blanco</h3>", unsafe_allow_html=True)   
st.markdown("<h4 style='text-align: center;'>Abstract</h3>", unsafe_allow_html=True)   
st.markdown("<div style='text-align: center;'><body>Text</body></div>", unsafe_allow_html=True)  

def about_description(DICT, i):
    cols = st.columns([5,1], gap='large', vertical_alignment='center')
    with cols[0]:    
        st.markdown(f"<h4 style='text-align: center;'>{DICT[i]['name']}</h4>", unsafe_allow_html=True)

        st.markdown(f"<div style='text-align: center;'><body>{DICT[i]['description']}</body></div>", unsafe_allow_html=True)  

    with cols[1]:
        st.image(DICT[i]['image'], use_container_width=True)

#---------------------------------#
st.markdown('#')
st.divider()
st.markdown("<h1 style='text-align: center;'>About us</h1>", unsafe_allow_html=True)   
st.divider()
about_description(ABOUT_DICT, 1)

add_footer()