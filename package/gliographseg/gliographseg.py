import os
import numpy as np
import torch
import networkx as nx

from PIL import Image
from skimage import segmentation, measure, graph
from torch_geometric.utils import from_networkx
from .model import SegmentGNN, predict

class GlioGraphSeg:
    """
    Graph-based image segmentation using Superpixels + GNN
    """

    def __init__(
        self,
        model_class,
        model_path=None,
        in_channels=3,
        hidden_channels=512,
        scale=1,
        sigma=0.8,
        min_size=20,
        device=None
    ):
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size

        self.device = device or self._identify_device()

        # 👇 MODEL PATH DI DEFAULT
        self.model_path = model_path or self._default_model_path()

        self.model = self._load_model(
            model_class,
            self.model_path,
            in_channels,
            hidden_channels
        )

    # --------------------------------------------------
    # DEFAULT MODEL PATH
    # --------------------------------------------------
    def _default_model_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "segmentgnn.pth")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Default model not found at {path}"
            )
        return path

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------
    def _identify_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    def _load_model(self, model_class, model_path, in_channels, hidden_channels):
        model = SegmentGNN(in_channels, hidden_channels)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model = model.to(torch.float32)
        model.eval()
        return model

    # --------------------------------------------------
    # SUPERPIXELS
    # --------------------------------------------------
    def _create_segments(self, image):
        return segmentation.felzenszwalb(
            image,
            scale=self.scale,
            sigma=self.sigma,
            min_size=self.min_size
        ) + 1

    # --------------------------------------------------
    # RAG
    # --------------------------------------------------
    def _create_rag(self, image, segments):
        rag = graph.rag_mean_color(image, segments, mode="similarity")
        G = nx.Graph()

        for node in rag.nodes:
            G.add_node(node, x=list(rag.nodes[node]["mean color"]))

        for u, v in rag.edges:
            G.add_edge(u, v)

        data = from_networkx(G)
        centers = [p.centroid for p in measure.regionprops(segments)]
        return data, centers, G

    # --------------------------------------------------
    # MASK
    # --------------------------------------------------
    def _create_mask(self, segments, preds):
        mask = np.zeros_like(segments, dtype=np.uint8)
        for seg_id in np.unique(segments):
            val = preds[seg_id - 1] if seg_id - 1 < len(preds) else 0
            mask[segments == seg_id] = int(val > 0.5)
        return mask

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    @torch.no_grad()
    def predict_mask(self, image):
        if isinstance(image, str):
            image = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image = np.array(image)

        segments = self._create_segments(image)
        data, centers, G = self._create_rag(image, segments)

        data = data.to(self.device)

        preds = predict(self.device, self.model, data)
        mask = self._create_mask(segments, preds)

        return {
            "mask": mask,
            "segments": segments,
            "graph": G,
            "centers": centers
        }

    def save_mask(self, mask, path):
        Image.fromarray(mask * 255).save(path)