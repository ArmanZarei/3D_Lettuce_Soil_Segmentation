import torch
import torch.nn as nn
try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_func=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.bn = nn.BatchNorm2d(
            out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_func = activation_func

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation_func:
            x = self.activation_func(x)

        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.device = device
        self.mlp = SharedMLP(10, d, bn=True, activation_func=nn.ReLU())

    def forward(self, coords, features, knn_output):
        idx, dist = knn_output
        B, N, K = idx.size()
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -
                                           1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)

        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)

        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )

        self.mlp = SharedMLP(in_channels, out_channels,
                             bn=True, activation_func=nn.ReLU())

    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        features = torch.sum(scores*x, dim=-1, keepdim=True)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(
            d_in, d_out//2, activation_func=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2*d_out)

        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        knn_output = knn(coords.cpu().contiguous(),
                         coords.cpu().contiguous(), self.num_neighbors)
        x = self.mlp1(features)
        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)
        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors, decimation, device):
        super(RandLANet, self).__init__()

        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.device = device

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_func=nn.ReLU())

        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_func=nn.ReLU(),
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_func=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_func=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

    def forward(self, input):
        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].clone().cpu()
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)

        decimation_ratio = 1

        # --------------- Encoder --------------- #
        x_stack = []
        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            x = lfa(coords[:, :N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, :N//decimation_ratio]
        # --------------------------------------- #

        x = self.mlp(x)

        # --------------- Decoder --------------- #
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:, :N//decimation_ratio].cpu().contiguous(),
                coords[:, :d*N//decimation_ratio].cpu().contiguous(),
                1
            )
            neighbors = neighbors.to(self.device)
            extended_neighbors = neighbors.unsqueeze(
                1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)
            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d
        # --------------------------------------- #

        x = x[:, :, torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)