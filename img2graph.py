import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
import skimage.segmentation as segmentation
import skimage.graph as graph
from skimage import measure


def create_segments(image,scale,sigma,min_size):
    segments = segmentation.felzenszwalb(image,scale=scale,sigma=sigma,min_size=min_size)
    segments = segments + 1  # Adjust the offset as needed
    boundaries = segmentation.mark_boundaries(image, segments, color=(1, 1, 0))  # Yellow boundaries
    return segments, boundaries


def craete_rag(image, segments):
    """
    Creates a Region Adiajency Graph (RAG) from the input image and its corresponding superpixel segments.

    Parameters:
    image (np.ndarray): The input image as a NumPy array.
    segments (np.ndarray): The superpixel segments as a NumPy array.

    Returns:
    Tuple[torch.Tensor, List[Tuple[int, int]], nx.Graph]: A tuple containing the RAG data, a list of centers for the superpixels, and the corresponding graph as a NetworkX object.

    This function first creates a RAG using the `graph.rag_mean_color` function from the `skimage.segmentation` module. It then constructs a graph using the NetworkX library, where each node represents a superpixel and each edge represents a connection between two superpixels. The `features` attribute of each node in the graph is set to the mean color of the corresponding superpixel. Finally, the function returns the RAG data, a list of centers for the superpixels, and the corresponding graph as a NetworkX object.
    """
    rag = graph.rag_mean_color(image, segments, mode='similarity')
    G = nx.Graph()
    

    for node in rag.nodes:
        mean_color = rag.nodes[node]['mean color']
        G.add_node(node, features=list(mean_color))

    for edge in rag.edges:
        node1, node2 = edge
        G.add_edge(node1, node2)


    data=from_networkx(G)
    if data.edge_index.max() >= data.num_nodes:
        raise ValueError(f"Edge index out of bounds: max index {data.edge_index.max()} exceeds number of nodes {data.num_nodes}")

    centers = []
    properties = measure.regionprops(segments)
    for p in properties:
        centers.append(p.centroid)
    return data,centers,G





    
def create_graph(image,scale,sigma,min_size):
    """
    Creates a Region Adjacency Graph (RAG) from an input MRI image and its corresponding mask, using superpixels and a specified SLIC algorithm parameters.

    Parameters:
    img_path (str): The path to the MRI image file.
    mask_path (str): The path to the mask file.
    ns (int): The desired number of superpixels to create.
    c (int): The compactness parameter for the SLIC algorithm, which controls the size and shape of the superpixels.
    sigma (float): The Gaussian smoothing parameter for the SLIC algorithm, which affects the spatial coherence of the superpixels.
    plot (bool, optional): A flag indicating whether to display the plot of the original image, color mask (segmentation image), superpixel boundaries image, and graph visualization image in a 2x2 grid. Defaults to False.

    Returns:
    RAGGraph: The Region Adjacency Graph (RAG) object containing the superpixel segments, labels, mask, and graph data.

    This function loads the MRI image and its corresponding mask, creates superpixels using the SLIC algorithm, assigns labels to the superpixels based on the most common label in the corresponding mask region, creates a RAG from the input image and its corresponding superpixel segments, sets the labels, mask, and segments attributes of the RAGGraph object, and optionally displays the plot.
    """
    # Load the input MRI image and its corresponding mask
    image = np.array(image)

    # Create superpixels from the input image using the SLIC algorithm
    segments, boundaries = create_segments(image,scale,sigma,min_size)


    # Create a Region Adjacency Graph (RAG) from the input image and its corresponding superpixel segments
    rag_graph, centers, nx_graph = craete_rag(image, segments)

    # Set the labels, mask, and segments attributes of the RAGGraph object
    rag_graph.segments = torch.tensor(segments)


    # If the plot parameter is set to True, display the plot of the original image, color mask (segmentation image), superpixel boundaries image, and graph visualization image in a 2x2 grid

    return rag_graph
