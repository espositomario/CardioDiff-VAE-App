from utils.my_module import *




with st.sidebar:
    download_table(DATA)



#-------------------Filter data by genes (rows)-------------------#
st.markdown("<h3 style='text-align: center;'>Filter data by genes (rows)</h3>", unsafe_allow_html=True)

#with st.expander("Select or Upload a gene list"):
SEL_GENES = select_genes()
# Plot Sankey diagram 
if SEL_GENES:
    # filter DATA by sekected genes
    st.dataframe(DATA.loc[SEL_GENES])
    
    #l
    
    
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Features distributions</h3>", unsafe_allow_html=True,help="..")

    C = st.columns(4, gap="small")

    for i,feature in enumerate(['RNA', 'H3K4me3', 'H3K27ac', 'H3K27me3']):
        FILT_DF = DATA[[f"{feature}_{ct}" for ct in CT_LIST]].copy()
        VMIN, VMAX = FILT_DF.min().min(), FILT_DF.max().max()
        VMIN = FILT_DF.quantile([0.001]).min(axis=1)[0.001]
        VMAX = FILT_DF.quantile([0.999]).max(axis=1)[0.999]
        FILT_DF = FILT_DF.loc[SEL_GENES]
        
        with C[i]:
            fig, ax= plot_violin_box(FILT_DF, feature, CT_LIST, HM_COL_DICT, CT_COL_DICT, y_lab='Z-score' if i==0 else'',
                                        VMIN=VMIN, VMAX=VMAX)

            st.pyplot(fig)
    
    
    
    # Plot Sankey diagram
    st.markdown("<h5 style='text-align: center;'>Gene to Cluster Sankey Diagram</h5>", unsafe_allow_html=True)
    fig = plot_sankey(DATA, SEL_GENES, font_color="white", font_size=14, link_opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)


#-------------------Filter data by features (columns)-------------------#
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Filter data by features (columns)</h3>", unsafe_allow_html=True)
#with st.expander("Select Features to display"):
df_tabs(DATA)


