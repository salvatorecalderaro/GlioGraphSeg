import streamlit as st
import numpy as np
from PIL import Image
import io
import zipfile
import os 
import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from skimage import segmentation, measure, graph
import platform
import cpuinfo
from model import SegmentGNN,predict
from huggingface_hub import hf_hub_download


os.environ["STREAMLIT_WATCH_MODE"] = "false"
LOGO_PATH = "images/GlioGraphSeg_logo.png"



scale=1
sigma=0.8
min_size=20
in_channels=3
hidden_channels=512
nclasses=1

def identify_device():
    so = platform.system()
    if so == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name

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


def load_model(device):
    model = SegmentGNN(in_channels, hidden_channels, 2).to(device)
    model_path = hf_hub_download(
        repo_id="salvatorecalderarp/GlioGraphSeg",  # Replace with your actual HF repo
        filename="model.pth"                      # Ensure this is the correct file name
    )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def build_mask_from_pred(segments,predicted_mask):
    num_mask = np.zeros_like(segments, dtype=np.int32)
    unique_segments = np.unique(segments)

    for segment_id in unique_segments:
        mask = (segments == segment_id)
        label = predicted_mask[segment_id - 1] if segment_id - 1 < len(predicted_mask) else 0
        num_mask[mask] = label

    return num_mask

# Placeholder segmentation function (replace with real GNN model)
@st.cache_data
def create_mask(graph, predictions):
    segments = graph.segments.detach().cpu().numpy()
    pred_mask = build_mask_from_pred(segments, predictions)  # NumPy array

    # Convert to image (for visualization and download)
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).convert("L")

    return mask_img, pred_mask

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

device,devname=identify_device()
model = load_model(device)

uploaded_file = st.file_uploader("Upload a glioma MRI image (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        graph = create_graph(image,scale,sigma,min_size)
        print(graph)
        if st.button("🩻 Segment Image"):
            with st.spinner("Segmenting image..."):
                predictions = predict(device, model, graph)
                mask_img,pred_mask = create_mask(graph,predictions)
                st.image(mask_img, caption="🧠 Segmented Output (Mask Only)", use_container_width=True)

                # Optional: overlay the mask on the original image
                overlay = Image.blend(image, mask_img.convert("RGB"), alpha=0.4)
                st.image(overlay, caption="🔍 Overlay of Mask on Original Image", use_container_width=True)

                # Prepare segmented image for download
                buf_img = io.BytesIO()
                mask_img.save(buf_img, format="PNG")
                byte_img = buf_img.getvalue()

                # Prepare NumPy mask for download
                buf_mask = io.BytesIO()
                np.save(buf_mask, pred_mask)
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