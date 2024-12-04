from utils.my_module import *
#st.set_page_config(layout="wide", initial_sidebar_state="expanded")



df_tabs(DATA)


def nodes_xy(len_gene_nodes, len_cluster_nodes):
    """
    Generate x and y positions for Sankey nodes.

    Parameters:
    - len_gene_nodes (int): Number of gene nodes.
    - len_cluster_nodes (int): Number of cluster nodes.

    Returns:
    - x (list): x positions for all nodes.
    - y (list): y positions for all nodes.
    """
    if len_gene_nodes < 2 :
        return None, None

    # x positions: fixed for gene nodes and varying for cluster nodes
    x_gene = [None] * len_gene_nodes
    x_cluster = list(np.linspace(0.3, 0.8, len_cluster_nodes))

    # y positions: linearly spaced from 0 to 1
    y_gene = [None] * len_gene_nodes
    y_cluster = list(np.linspace(0.1, 0.95, len_cluster_nodes))

    # Combine x and y positions
    x = x_gene + x_cluster
    y = y_gene + y_cluster

    return x, y





import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go

def plot_sankey(DATA, SEL_GENES, font_size=12, font_color="black", link_opacity=0.5):
    """
    Create a Sankey diagram with genes as the first layer and clusters as the second.
    Links between genes and clusters are colored based on clusters.

    Parameters:
    - DATA (pd.DataFrame): Input data with genes as index.
    - SEL_GENES (list): List of selected gene names.
    - font_size (int): Font size for node labels.
    - font_color (str): Font color for node labels.
    - font_family (str): Font family for node labels.
    - link_opacity (float): Opacity for the links (0.0 to 1.0).

    Returns:
    - fig: A Plotly Sankey figure.
    """
    # Filter for selected genes
    data_filtered = DATA.loc[SEL_GENES]

    # Create node labels
    gene_nodes = SEL_GENES  # Gene names as the first level
    cluster_nodes = data_filtered["Cluster"].unique().tolist()  # Clusters as the second level
    
    all_nodes = gene_nodes + cluster_nodes  # Combine all nodes
    node_map = {node: i for i, node in enumerate(all_nodes)}  # Map node name to index

    # Generate a colormap for clusters
    num_clusters = len(cluster_nodes)
    cmap = cm.get_cmap("rainbow", num_clusters)  # Use rainbow colormap
    cluster_colors = {cluster: mcolors.rgb2hex(cmap(i)) for i, cluster in enumerate(cluster_nodes)}

    # Create links
    links = []
    link_colors = []

    # Links from genes to clusters
    for gene, row in data_filtered.iterrows():
        cluster = row["Cluster"]
        rgba_color = mcolors.to_rgba(cluster_colors[cluster], alpha=link_opacity)  # Convert hex to RGBA
        rgba_str = f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})"
        links.append({
            "source": node_map[gene],
            "target": node_map[cluster],
            "value": 1  # Equal weight for all links
        })
        link_colors.append(rgba_str)

    # Define node colors
    node_colors = []
    for node in all_nodes:
        if node in gene_nodes:  # Gene nodes
            node_colors.append("silver")  # Default gray for genes
        else:  # Cluster nodes
            node_colors.append(cluster_colors[node])  # Color by cluster

    # Create Sankey diagram
    fig = go.Figure(go.Sankey(
        textfont=dict(size=font_size, color=font_color),
        #arrangement = "freeform",
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            x=nodes_xy(len(gene_nodes), len(cluster_nodes))[0],
            y=nodes_xy(len(gene_nodes), len(cluster_nodes))[1],
            color=node_colors,
            hovertemplate='%{label}<extra></extra>',
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            color=link_colors,  # Colored links based on clusters
            hovertemplate='Gene: %{source.label}<br>Cluster: %{target.label}<extra></extra>',
        )
    ))

    # Update layout
    fig.update_layout(
        title_text="",
        margin=dict(t=50, l=0, r=25, b=25),
        height=  70+ (len(gene_nodes) * 25),  # Set a fixed height

    )
    
    return fig





MARKER_GENES=['Nanog','Pou5f1','Sox2','Dppa5a','Mesp1','T', 'Vrtn','Dll3','Gata5', 'Tek','Sox18','Lyl1','Actn2', 'Coro6','Myh6','Myh7']
CAT_FEATURE = 'CV_Category'
COL_DICT =  {'RNA_ESC': '#405074',
                'RNA_MES': '#7d5185',
                'RNA_CP': '#c36171',
                'RNA_CM': '#eea98d',
                'STABLE':'#B4CD70',
                'other':'#ECECEC'}



SEL_GENES= st.multiselect("Select genes", options=DATA.index, default=MARKER_GENES,placeholder="Select genes",
                #max_selections=20,
                key="select a gene")

if SEL_GENES:
    st.markdown("<h3 style='text-align: center;'>Gene to Cluster Sankey diagram</h3>", unsafe_allow_html=True)

    fig = plot_sankey(DATA, SEL_GENES,font_color='white', font_size=16,link_opacity=0.5)
    st.plotly_chart(fig, use_container_width=True,)

