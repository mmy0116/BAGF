import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Sigmoid, Dropout
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dropout_edge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BAttention(torch.nn.Module):
    

    def __init__(self, hidden_channels):
        super().__init__()

        self.query1 = Linear(hidden_channels, hidden_channels)
        self.key1 = Linear(hidden_channels, hidden_channels)
        self.value1 = Linear(hidden_channels, hidden_channels)

        self.query2 = Linear(hidden_channels, hidden_channels)
        self.key2 = Linear(hidden_channels, hidden_channels)
        self.value2 = Linear(hidden_channels, hidden_channels)

        self.out_proj1 = Linear(hidden_channels, hidden_channels)
        self.out_proj2 = Linear(hidden_channels, hidden_channels)

        self.scale = torch.sqrt(torch.tensor(hidden_channels, dtype=torch.float32))

    def forward(self, x1, x2):
        q1 = self.query1(x1)
        k2 = self.key2(x2)
        v2 = self.value2(x2)

        attn1 = torch.matmul(q1, k2.transpose(-2, -1)) / self.scale
        attn1 = F.softmax(attn1, dim=-1)

        out1 = torch.matmul(attn1, v2)
        out1 = self.out_proj1(out1)

        q2 = self.query2(x2)
        k1 = self.key1(x1)
        v1 = self.value1(x1)

        attn2 = torch.matmul(q2, k1.transpose(-2, -1)) / self.scale
        attn2 = F.softmax(attn2, dim=-1)

        out2 = torch.matmul(attn2, v1)
        out2 = self.out_proj2(out2)

        fused1 = x1 + out1
        fused2 = x2 + out2

        return fused1, fused2

class GateFusion(torch.nn.Module):
    

    def __init__(self, hidden_channels):
        super().__init__()
        self.gate = Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, 1),
            Sigmoid()
        )

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        alpha = self.gate(z)
        fused = alpha * x + (1 - alpha) * y
        return fused

class my(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels
        out_channels = self.args.out_channels
        K = 2

        self.linear1_branch1 = Linear(in_channels, hidden_channels)
        self.linear1_branch2 = Linear(in_channels, hidden_channels)

        self.conv1_branch1 = ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym")
        self.conv2_branch1 = ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym")
        self.conv3_branch1 = ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym")

        self.conv1_branch2 = ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym")
        self.conv2_branch2 = ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym")
        self.conv3_branch2 = ChebConv(hidden_channels, hidden_channels, K=K, normalization="sym")

        self.cross_attn1 = BAttention(hidden_channels)
        self.cross_attn2 = BAttention(hidden_channels)

        self.fusion3 = GateFusion(hidden_channels)

        self.classifier = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Dropout(p=0.6),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        PPI_input = data[0].x
        edge_index_PPI = data[0].edge_index
        edge_index_PPI, _ = dropout_edge(edge_index_PPI, p=0.3, force_undirected=True, training=self.training)
        Path_input = data[1].x
        edge_index_Path = data[1].edge_index
        edge_index_Path, _ = dropout_edge(edge_index_Path, p=0.3, force_undirected=True, training=self.training)

        PPI_input = F.dropout(PPI_input, p=0.3, training=self.training)
        Path_input = F.dropout(Path_input, p=0.3, training=self.training)

        x1 = F.relu(self.linear1_branch1(PPI_input))
        x1 = F.dropout(x1, p=0.3, training=self.training)

        x2 = F.relu(self.linear1_branch2(Path_input))
        x2 = F.dropout(x2, p=0.3, training=self.training)

        x1 = self.conv1_branch1(x1, edge_index_PPI)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.conv1_branch2(x2, edge_index_Path)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)

        x1, x2 = self.cross_attn1(x1, x2)
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x2 = F.dropout(x2, p=0.3, training=self.training)

        x1 = self.conv2_branch1(x1, edge_index_PPI)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.45, training=self.training)

        x2 = self.conv2_branch2(x2, edge_index_Path)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.45, training=self.training)

        x1, x2 = self.cross_attn2(x1, x2)
        x1 = F.dropout(x1, p=0.35, training=self.training)
        x2 = F.dropout(x2, p=0.35, training=self.training)

        x1_final = self.conv3_branch1(x1, edge_index_PPI)
        x1_final = F.relu(x1_final)
        x1_final = F.dropout(x1_final, p=0.5, training=self.training)

        x2_final = self.conv3_branch2(x2, edge_index_Path)
        x2_final = F.relu(x2_final)
        x2_final = F.dropout(x2_final, p=0.5, training=self.training)

        fused_final = self.fusion3(x1_final, x2_final)
        fused_final = F.dropout(fused_final, p=0.55, training=self.training)

        out = self.classifier(fused_final)

        return out

