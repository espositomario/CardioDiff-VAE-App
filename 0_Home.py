from utils.my_module import *

st.markdown(f"<h1 style='text-align: center;'>{APP_NAME}</h3>", unsafe_allow_html=True)   
st.divider()


st.markdown(f"<h3 style='text-align: center;'>Functions</h3>", unsafe_allow_html=True)   

cols = st.columns(4, gap='large', vertical_alignment='bottom')
for i in range(4):
    with cols[i]:
        page_cover(PAGES_DICT, i+1)

st.divider()
#---------------------------------#
def page_description(PAGES_DICT, i):
    cols = st.columns([3,1], gap='large', vertical_alignment='center')
    with cols[0]:    
        st.markdown(f"<h3 style='text-align: center;'>{PAGES_DICT[i]['name']}</h3>", unsafe_allow_html=True)

        st.markdown(f"<div style='text-align: center;'><body>{PAGES_DICT[i]['description']}</body></div>", unsafe_allow_html=True)  

    with cols[1]:
        st.image(PAGES_DICT[i]['image'], use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    
for i in range(4):
    page_description(PAGES_DICT, i+1)








    
add_footer()