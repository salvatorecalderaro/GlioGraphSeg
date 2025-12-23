import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# SegmentGNN with Dropout and adjusted loss function
class SegmentGNN(torch.nn.Module):
    def __init__(self, in_features, hidden_channels):
        super(SegmentGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # Batch normalization layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x,edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        return x  # No Sigmoid here, use BCEWithLogitsLoss



def predict(device, model, data):
    model.eval()
    # send your data to device
    x, edge_index = data.features, data.edge_index
    if device == torch.device("mps"):
        x = x.to(torch.float32).to(device)
    else :
        x = x.float().to(device)
    edge_index = edge_index.to(device)
    with torch.no_grad():
        logits = model(x, edge_index).squeeze(-1)   # [N]
        probs  = torch.sigmoid(logits)     
        preds  = (probs > 0.5).long().cpu().numpy() # [N]
    return preds