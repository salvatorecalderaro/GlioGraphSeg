import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# SegmentGNN with Dropout and adjusted loss function
class SegmentGNN(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes):
        super(SegmentGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # Batch normalization layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.features.float(), data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.conv2(x, edge_index)
        return x  # No Sigmoid here, use BCEWithLogitsLoss


# Custom Dice Loss Function
def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)  # Apply Sigmoid
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


# Combined BCEWithLogitsLoss and Dice Loss
class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, pos_weight=None):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        dice = dice_loss(logits, targets.float())
        return bce_loss + dice



def predict(device, model, testloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            data = data.to(device)
            out = model(data)
            out = torch.round(torch.sigmoid(out))  # Apply Sigmoid for predictions
            predictions.append(out.detach().cpu().numpy())
    return predictions
    

