import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.conv3 = SAGEConv(hidden_channels, num_features, aggr="mean")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)  

        x = self.conv3(x, edge_index)
        x = x.relu() 
        return x


class ResidualFusion(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=0)
        )
 
        self.residual = nn.Linear(input_dim, input_dim)

    def forward(self, features):

        weights = self.attention(torch.stack(features))  # [4, N, 1]
        weighted = torch.sum(weights * torch.stack(features), dim=0)  # [N, 512]

        residual = self.residual(features[0])
        fused = weighted + residual
        return F.relu(fused)



class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
 
        self.Coefficient.data = (self.Coefficient.data + self.Coefficient.data.T) / 2
 
        self.Coefficient.data = F.softshrink(self.Coefficient.data, lambd=1e-5)
        y = torch.matmul(self.Coefficient, x)
        return y

class GraphSAGECluster(torch.nn.Module):
    def __init__(self, features, hidden_channels, num_sample):
        super(GraphSAGECluster, self).__init__()
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3], dtype=torch.float32))  
        self.gcnconv1 = GraphSAGE(features[0].shape[-1], hidden_channels)
        self.gcnconv2 = GraphSAGE(features[1].shape[-1], hidden_channels)
        self.gcnconv3 = GraphSAGE(features[2].shape[-1], hidden_channels)
        self.gcnconv4 = GraphSAGE(features[3].shape[-1], hidden_channels)

        self.lin1 = nn.Linear(features[0].shape[-1], 512)
        self.lin2 = nn.Linear(features[1].shape[-1], 512)
        self.lin3 = nn.Linear(features[2].shape[-1], 512)
        self.lin4 = nn.Linear(features[3].shape[-1], 512)

        self.residual_fusion = ResidualFusion(input_dim=512)

        self.content_expression = SelfExpression(num_sample)
        self.structure_expression = SelfExpression(num_sample)

        self.W = nn.Linear(2 * num_sample, 2)


    def forward(self, datas):
        x1 = self.gcnconv1(datas[0].x, datas[0].edge_index)
        x2 = self.gcnconv2(datas[1].x, datas[1].edge_index)
        x3 = self.gcnconv3(datas[2].x, datas[2].edge_index)
        x4 = self.gcnconv4(datas[3].x, datas[3].edge_index)

        feat1 = self.lin1(x1)
        feat2 = self.lin2(x2)
        feat3 = self.lin3(x3)
        feat4 = self.lin4(x4)

        structure_features = self.residual_fusion([feat1, feat2, feat3, feat4])
        content_features = datas[-1].x


        weights = F.softmax(self.fusion_weight, dim=0)
        fusion_expression = weights[0] * self.content_expression.Coefficient + weights[
            1] * self.structure_expression.Coefficient

        return fusion_expression, content_features, structure_features, self.content_expression.Coefficient, self.structure_expression.Coefficient
