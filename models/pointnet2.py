import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d
from torch_geometric.nn import PointConv, fps, radius, knn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data.data import Data
from torch_scatter import scatter_add, scatter_max


class PointNet2SAModule(torch.nn.Module):
    def __init__(self, sample_radio, radius, max_num_neighbors, mlp):
        super(PointNet2SAModule, self).__init__()
        self.sample_ratio = sample_radio
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.point_conv = PointConv(mlp)

    def forward(self, data):
        x, pos, batch = data

        # Sample
        idx = fps(pos, batch, ratio=self.sample_ratio)

        # Group(Build graph)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=self.max_num_neighbors)
        edge_index = torch.stack([col, row], dim=0)

        # Apply pointnet
        x1 = self.point_conv(x, (pos, pos[idx]), edge_index)
        pos1, batch1 = pos[idx], batch[idx]

        return x1, pos1, batch1


class PointNet2GlobalSAModule(torch.nn.Module):
    '''
    One group with all input points, can be viewed as a simple PointNet module.
    It also return the only one output point(set as origin point).
    '''
    def __init__(self, mlp):
        super(PointNet2GlobalSAModule, self).__init__()
        self.mlp = mlp

    def forward(self, data):
        x, pos, batch = data
        if x is not None: x = torch.cat([x, pos], dim=1)
        x1 = self.mlp(x)

        x1 = scatter_max(x1, batch, dim=0)[0]  # (batch_size, C1)

        batch_size = x1.shape[0]
        pos1 = x1.new_zeros((batch_size, 3))  # set the output point as origin
        batch1 = torch.arange(batch_size).to(batch.device, batch.dtype)

        return x1, pos1, batch1


class PointConvFP(MessagePassing):
    '''
    Core layer of Feature propagtaion module.
    '''
    def __init__(self, mlp=None):
        super(PointConvFP, self).__init__('add', 'source_to_target')
        self.mlp = mlp
        self.aggr = 'add'
        self.flow = 'source_to_target'

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, pos, edge_index):
        r"""
        Args:
            x (tuple), (tensor, tensor) or (tensor, NoneType)
            pos (tuple): The node position matrix. Either given as
                tensor for use in general message passing or as tuple for use
                in message passing in bipartite graphs.
            edge_index (LongTensor): The edge indices.
        """
        # Do not pass (tensor, None) directly into propagate(), sice it will check each item's size() inside.
        x_tmp = x[0] if x[1] is None else x
        aggr_out = self.propagate(edge_index, x=x_tmp, pos=pos)

        #
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        x_target, pos_target = x[i], pos[i]

        add = [pos_target,] if x_target is None else [x_target, pos_target]
        aggr_out = torch.cat([aggr_out, *add], dim=1)

        if self.mlp is not None: aggr_out = self.mlp(aggr_out)

        return aggr_out

    def message(self, x_j, pos_j, pos_i, edge_index):
        '''
        x_j: (E, in_channels)
        pos_j: (E, 3)
        pos_i: (E, 3)
        '''
        dist = (pos_j - pos_i).pow(2).sum(dim=1).pow(0.5)
        dist = torch.max(dist, torch.Tensor([1e-10]).to(dist.device, dist.dtype))
        weight = 1.0 / dist  # (E,)

        row, col = edge_index
        index = col
        num_nodes = maybe_num_nodes(index, None)
        wsum = scatter_add(weight, col, dim=0, dim_size=num_nodes)[index] + 1e-16  # (E,)
        weight /= wsum

        return weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class PointNet2FPModule(torch.nn.Module):
    def __init__(self, knn_num, mlp):
        super(PointNet2FPModule, self).__init__()
        self.knn_num = knn_num
        self.point_conv = PointConvFP(mlp)

    def forward(self, in_layer_data, skip_layer_data):
        in_x, in_pos, in_batch = in_layer_data
        skip_x, skip_pos, skip_batch = skip_layer_data

        row, col = knn(in_pos, skip_pos, self.knn_num, in_batch, skip_batch)        
        edge_index = torch.stack([col, row], dim=0)

        x1 = self.point_conv((in_x, skip_x), (in_pos, skip_pos), edge_index)
        pos1, batch1 = skip_pos, skip_batch 

        return x1, pos1, batch1


def make_mlp(in_channels, mlp_channels, batch_norm=True):
    assert len(mlp_channels) >= 1
    layers = []

    for c in mlp_channels:
        layers += [Lin(in_channels, c)]
        if batch_norm: layers += [BatchNorm1d(c)]
        layers += [ReLU()]

        in_channels = c

    return Seq(*layers)


class PointNet2(torch.nn.Module):
    '''
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    '''
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        self.num_classes = num_classes

        # SA1
        sa1_sample_ratio = 0.5
        sa1_radius = 0.2
        sa1_max_num_neighbours = 64
        sa1_mlp = make_mlp(3, [64, 64, 128])
        self.sa1_module = PointNet2SAModule(sa1_sample_ratio, sa1_radius, sa1_max_num_neighbours, sa1_mlp)

        # SA2
        sa2_sample_ratio = 0.25
        sa2_radius = 0.4
        sa2_max_num_neighbours = 64
        sa2_mlp = make_mlp(128+3, [128, 128, 256])
        self.sa2_module = PointNet2SAModule(sa2_sample_ratio, sa2_radius, sa2_max_num_neighbours, sa2_mlp)

        # SA3
        sa3_mlp = make_mlp(256+3, [256, 512, 1024])
        self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        ##
        knn_num = 3

        # FP3, reverse of sa3
        fp3_knn_num = 1  # After global sa module, there is only one point in point cloud
        fp3_mlp = make_mlp(1024+256+3, [256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256+128+3, [256, 128])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        fp1_mlp = make_mlp(128+3, [128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

        self.fc1 = Lin(128, 128)
        self.dropout1 = Dropout(p=0.5)
        self.fc2 = Lin(128, self.num_classes)

    def forward(self, data):
        '''
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input
            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes 
                data.pos: (batch_size * ~num_points, 3)
                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        '''
        data = data.transpose(1, 2)
        dense_input = True if isinstance(data, torch.Tensor) else False

        if dense_input:
            # Convert to torch_geometric.data.Data type
            data = data.transpose(1, 2).contiguous()
            batch_size, N, _ = data.shape  # (batch_size, num_points, 3)
            pos = data.view(batch_size*N, -1)
            batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
            for i in range(batch_size): batch[i] = i
            batch = batch.view(-1)

            data = Data()
            data.pos, data.batch = pos, batch

        if not hasattr(data, 'x'): data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)

        fp3_out = self.fp3_module(sa3_out, sa2_out)
        fp2_out = self.fp2_module(fp3_out, sa1_out)
        fp1_out = self.fp1_module(fp2_out, data_in)

        fp1_out_x, fp1_out_pos, fp1_out_batch = fp1_out
        x = self.fc2(self.dropout1(self.fc1(fp1_out_x)))
        x = F.log_softmax(x, dim=-1)

        if dense_input: return x.view(batch_size, N, self.num_classes).transpose(1, 2)
        else: return x, fp1_out_batch