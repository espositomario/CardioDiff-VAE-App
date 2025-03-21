from utils.my_module import *

st.divider()
st.markdown("<h2 style='text-align: center;'>TITLE</h2>", unsafe_allow_html=True)   
st.markdown("<h3 style='text-align: center;'>Mario Esposito, Luciano Di Croce and Enrique Blanco</h3>", unsafe_allow_html=True)   
st.markdown("<h4 style='text-align: center;'>Abstract</h4>", unsafe_allow_html=True)   
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



#---------------------------------#
st.markdown('#')
st.divider()
st.markdown("<h1 style='text-align: center;'>Credits and References</h1>", unsafe_allow_html=True)   
st.divider()
CREDITS= """
- **Streamlit**  [Official Website](https://streamlit.io/) 

- **Plotly**  [Official Website](https://plotly.com/) 

- **Pandas**  [Official Website](https://pandas.pydata.org/) McKinney, W. (2010)

- **Streamlit PDF Viewer**  [GitHub Repository](https://github.com/andfanilo/streamlit-pdf-viewer)

- **Streamlit Extras**  [GitHub Repository](https://arnaudmiribel.github.io/streamlit-extras/)

- **Streamlit Searchbox** [GitHub Repository](https://github.com/m-wrzr/streamlit-searchbox)

- **Matplotlib**  [Official Website](https://matplotlib.org/) Hunter, J. D. (2007)

- **Seaborn** [Official Website](https://seaborn.pydata.org/) Waskom, M. L. (2021)

- **Kaleido**  [GitHub Repository](https://github.com/plotly/Kaleido)

- **PyMuPDF (fitz)**  [GitHub Repository](https://github.com/pymupdf/PyMuPDF)

- **GSEApy**  [Official Website](https://gseapy.readthedocs.io/en/latest/) Zhuoqing, F. (2023)
  
"""

st.markdown(CREDITS)





add_footer()