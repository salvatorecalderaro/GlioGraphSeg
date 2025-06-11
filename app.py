import streamlit as st
import numpy as np
from PIL import Image
import io
import zipfile
from img2graph import craete_graph

# Path to your logo image
LOGO_PATH = "images/GlioGraphSeg_logo.png"

scale=1
sigma=0.8
min_size=20


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
        graph = craete_graph(image,scale,sigma,min_size)
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