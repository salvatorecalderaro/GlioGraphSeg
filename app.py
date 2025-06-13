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

os.environ["STREAMLIT_WATCH_MODE"] = "false"
LOGO_PATH = "images/GlioGraphSeg_logo.png"
MODEL_PATH = "model.pth"



scale=1
sigma=0.8
min_size=20
in_channels=3
hidden_channels=512
nclasses=1


def identify_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

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




def load_model(device):
    model = SegmentGNN(in_channels, hidden_channels).to(device)    # Load weights
    state_dict = torch.load(MODEL_PATH,map_location=device)
    model.load_state_dict(state_dict)
    if device == torch.device("mps"):
        model = model.to(torch.float32)
    else:
        model = model.float()
    return model


# Placeholder segmentation function (replace with real GNN model)

def create_mask(segments, predicted_mask):
    num_mask = np.zeros_like(segments, dtype=np.int32)
    unique_segments = np.unique(segments)
    
    for segment_id in unique_segments:
        mask = (segments == segment_id)
        label = predicted_mask[segment_id - 1] if segment_id - 1 < len(predicted_mask) else 0
        num_mask[mask] = label

    return num_mask
# Streamlit page configuration
st.set_page_config(page_title="GliomGraphSeg", page_icon=LOGO_PATH)

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

    **Citation:**  
    Amato, D., Calderaro, S., Lo Bosco, G., Rizzo, R., & Vella, F. (2024, December).  
    _Semantic Segmentation of Gliomas on Brain MRIs by Graph Convolutional Neural Networks_.  
    In 2024 International Conference on AI x Data and Knowledge Engineering (AIxDKE), IEEE.  
    [📄 Paper Link](https://ieeexplore.ieee.org/abstract/document/10990089/)

    ---
    """)

# Main app UI
st.title("🧠 Glioma Image Segmentation using GNN")
device = identify_device()
model = load_model(device)
model.eval()
uploaded_file = st.file_uploader("Upload a glioma MRI image (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tiff"])
st.markdown(
    "<span style='color:gray; font-size:0.9em;'>"
    "ℹ️ For best results, upload a **RGB image composed of three MRI modalities** in the following channel order: "
    "**Red = pre-contrast**, **Green = FLAIR**, **Blue = post-contrast**."
    "</span>",
    unsafe_allow_html=True
)
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
        segments, boundaries = create_segments(np.array(image),scale,sigma,min_size)
        data,centers,G = craete_rag(np.array(image),segments)
        data.segments = torch.tensor(segments)
        if st.button("🩻 Segment Image"):
            with st.spinner("Segmenting image..."):
                predictions = predict(device, model, data)          
                pred_mask = create_mask(data.segments, predictions)
                st.image((pred_mask * 255).astype(np.uint8), caption="🧠 Segmented Output (Mask Only)", use_container_width=True)
                mask_img = (pred_mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_img)
                # Optional: overlay the mask on the original image
                if mask_img.size != image.size:
                    mask_img = mask_img.resize(image.size, resample=Image.NEAREST)

                # Convert mask image to same mode as original image
                if mask_img.mode != image.mode:
                    mask_img = mask_img.convert(image.mode)

                
                overlay = Image.blend(image, mask_img, alpha=0.4)
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