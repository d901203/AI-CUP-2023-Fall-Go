import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class BatchRenorm2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("running_std", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.weight = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        """
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0)
                batch_std = z.std(0, unbiased=False) + self.eps
            else:
                batch_mean = x.mean(dims)
                batch_std = x.std(dims, unbiased=False) + self.eps

            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean)) / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_std += self.momentum * (batch_std.detach() - self.running_std)
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


eps = 1e-3
momentum = 1e-2


class GlobalPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.pool = GlobalPool()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1, padding=0, bias=True),
            nn.Mish(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, kernel_size=1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x)
        out = self.conv(out)
        gammas, betas = torch.split(out, self.channels, dim=1)
        gammas = torch.reshape(gammas, (b, c, 1, 1))
        betas = torch.reshape(betas, (b, c, 1, 1))
        out = self.sigmoid(gammas) * x + betas
        return out


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False):
        super().__init__()
        self.norm = BatchRenorm2d(in_channels, eps=eps, momentum=momentum)
        self.act = nn.Mish(inplace=True)
        self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.se = SEBlock(out_channels) if use_se else None

    def forward(self, x):
        out = x
        out = self.norm(out)
        out = self.act(out)
        if self.se is None:
            return self.conv_3x3(out) + self.conv_1x1(out)
        else:
            return self.se(out)


class InnerResidualBlock(nn.Module):
    def __init__(self, channels, use_se=False):
        super().__init__()

        self.conv1 = NormActConv(channels, channels, use_se=use_se)
        self.conv2 = NormActConv(channels, channels)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        return out + x


class NestedResidualBlock(nn.Module):
    def __init__(self, channels, use_se=False):
        super().__init__()

        c = channels
        c2 = c // 2

        self.conv_in = NormActConv(c, c2)

        self.inner_block1 = InnerResidualBlock(c2, use_se=use_se)
        self.inner_block2 = InnerResidualBlock(c2)

        self.conv_out = NormActConv(c2, c)

    def forward(self, x):
        out = x
        out = self.conv_in(out)
        out = self.inner_block1(out)
        out = self.inner_block2(out)
        out = self.conv_out(out)
        return out + x


class PolicyHead(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=False)
        self.norm = BatchRenorm2d(channels, eps=eps, momentum=momentum)
        self.act = nn.Mish(inplace=True)
        self.fc = nn.Linear(1 * 19 * 19, 3)

    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 256

        self.conv_in = nn.Conv2d(17, channels, kernel_size=3, padding=1, bias=False)

        self.blocks = []
        for _ in range(2):
            self.blocks += [
                NestedResidualBlock(channels),
                NestedResidualBlock(channels, use_se=True),
            ]
        self.blocks.append(NestedResidualBlock(channels))
        self.blocks = nn.Sequential(*self.blocks)

        self.policy_head = PolicyHead(channels)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.blocks(out)
        out = self.policy_head(out)
        return out


A = "public"  # public or private
MODEL = "E"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"./csv/play_style_test_{A}.csv", "r") as f:
        df = f.readlines()

    games_id = [i.split(",", 2)[0] for i in df]
    games = [i.split(",", 1)[-1] for i in df]

    X = np.load(f"./data/play_style/test/{A}.npy")
    X = torch.from_numpy(X).float().to(device)

    dataset = TensorDataset(X)
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = Model().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f"./weights/play_style/{MODEL}.pth"))
    model.eval()

    with open(f"./play_style_{MODEL}_{A}.csv", "w") as f:
        for i, (data,) in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                y_pred = model(data)
                indices = torch.topk(torch.softmax(y_pred, dim=1), k=1, dim=1)[1]
                y_pred = indices.cpu().numpy()

            start_idx = i * batch_size
            end_idx = start_idx + y_pred.shape[0]
            for game_id, preds in zip(games_id[start_idx:end_idx], y_pred):
                f.write(f"{game_id},{int(preds[0]) + 1}\n")
