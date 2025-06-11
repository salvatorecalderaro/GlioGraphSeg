import streamlit as st
import numpy as np
from PIL import Image
import io
import zipfile
import os 
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from skimage import segmentation, measure, graph

os.environ["STREAMLIT_WATCH_MODE"] = "false"


# Path to your logo image
LOGO_PATH = "images/GlioGraphSeg_logo.png"

scale=1
sigma=0.8
min_size=20

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



# Placeholder segmentation function (replace with real GNN model)
@st.cache_data
def segment_image(image):
    """
    Dummy segmentation using thresholding.
    Replace this with actual GNN model inference.
    """
    img_array = np.array(image.convert("L"))  # Convert to grayscale
    mask = (img_array > 128).astype(np.uint8) * 255  # Simple threshold
    segmented_img = Image.fromarray(mask)
    return segmented_img, mask

# Streamlit page configuration
st.set_page_config(page_title="GliomGraphSeg", page_icon="🧠")

# Sidebar with logo and reference
with st.sidebar:
    try:
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=200)
    except Exception as e:
        st.warning(f"⚠️ Could not load logo: {e}")

    st.markdown("""
    ---
    ## 📄 Reference
    
    This app uses a **Graph Neural Network (GNN)** to perform glioma segmentation from brain MRI images.
    
    **Citation**:  
    Amato, D., Calderaro, S., Lo Bosco, G., Rizzo, R., & Vella, F. (2024, December).  
    _Semantic Segmentation of Gliomas on Brain MRIs by Graph Convolutional Neural Networks_.  
    In 2024 International Conference on AI x Data and Knowledge Engineering (AIxDKE), IEEE.
    ---
    """)

# Main app UI
st.title("🧠 Glioma Image Segmentation using GNN")

uploaded_file = st.file_uploader("Upload a glioma MRI image (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        graph = create_graph(image,scale,sigma,min_size)
        print(graph)
        if st.button("🩻 Segment Image"):
            with st.spinner("Segmenting image..."):
                segmented_img, seg_mask = segment_image(image)

                st.image(segmented_img, caption="🧠 Segmented Output (Mask Only)", use_container_width=True)

                # Optional: overlay the mask on the original image
                overlay = Image.blend(image, segmented_img.convert("RGB"), alpha=0.4)
                st.image(overlay, caption="🔍 Overlay of Mask on Original Image", use_container_width=True)

                # Prepare segmented image for download
                buf_img = io.BytesIO()
                segmented_img.save(buf_img, format="PNG")
                byte_img = buf_img.getvalue()

                # Prepare NumPy mask for download
                buf_mask = io.BytesIO()
                np.save(buf_mask, seg_mask)
                buf_mask.seek(0)

                # Individual download buttons
                st.download_button(
                    label="📥 Download Segmented Image (PNG)",
                    data=byte_img,
                    file_name="segmented_image.png",
                    mime="image/png"
                )

                st.download_button(
                    label="📥 Download Segmentation Mask (.npy)",
                    data=buf_mask,
                    file_name="segmentation_mask.npy",
                    mime="application/octet-stream"
                )

                # Optional: ZIP download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    zipf.writestr("segmented_image.png", byte_img)
                    zipf.writestr("segmentation_mask.npy", buf_mask.getvalue())

                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download All Results (ZIP)",
                    data=zip_buffer,
                    file_name="segmentation_results.zip",
                    mime="application/zip"
                )

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")