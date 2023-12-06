from bisect import bisect
from collections import defaultdict
from datetime import datetime, timedelta
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from lion_pytorch import Lion
from torch.utils.data import DataLoader, random_split
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
    def __init__(self, channels, reduction=3):
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

        self.conv = nn.Conv2d(channels, 2, kernel_size=1, padding=0, bias=False)
        self.norm = BatchRenorm2d(channels, eps=eps, momentum=momentum)
        self.act = nn.Mish(inplace=True)
        self.fc = nn.Linear(2 * 19 * 19, 19 * 19)

    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 192

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


# Define the training step:
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, float]:
    # Set the model to train mode
    model.train()
    # Set the training loss to 0
    train_loss = train_accuracy = 0
    # Iterate over the DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
        # Move the batch to the device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # 1. Zero the gradients
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False):
            # 2. Forward pass
            y_pred = model(X)
            # 3. Calculate and accumulate loss
            loss = loss_func(y_pred, y)
        train_loss += loss.item()
        with torch.autograd.detect_anomaly():
            # 4. Backward pass
            scaler.scale(loss).backward()
        # 5. Update the optimizer
        scaler.step(optimizer)
        scaler.update()
        # 6. Calculate the accuracy
        topk_preds = torch.topk(y_pred, 1, dim=1)[1]
        correct = topk_preds.eq(y.view(-1, 1).expand_as(topk_preds)).sum().item()
        train_accuracy += correct / len(y)
    # Return average loss
    return train_loss / len(dataloader), train_accuracy / len(dataloader)


# Define the validating step:
# Turn on inference context manager
@torch.no_grad()
def valid_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    # Set the model to eval mode
    model.eval()
    # Set the validating loss to 0
    valid_loss = valid_accuracy = 0
    # Iterate over the DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
        # Move the batch to the device
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate and accumulate loss
        valid_loss += loss_func(y_pred, y).item()
        # 3. Calculate the accuracy
        topk_preds = torch.topk(y_pred, 1, dim=1)[1]
        correct = topk_preds.eq(y.view(-1, 1).expand_as(topk_preds)).sum().item()
        valid_accuracy += correct / len(y)
    # Return average loss
    return valid_loss / len(dataloader), valid_accuracy / len(dataloader)


# Define the training and validating loops:
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epochs: int,
    device: torch.device,
) -> defaultdict:

    # Init the results
    result = defaultdict(list)
    # Set the model to the device
    model.to(device)
    best_accuracy = 0
    best_loss = float("inf")
    patience = 2
    early_stopping_counter = 0
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    elapsed_time = 0

    # Iterate over the epochs
    for epoch in tqdm(range(1, epochs + 1)):
        start_time = datetime.now()
        print(f"\n{start_time.strftime('%Y/%m/%d %H:%M:%S')} ({elapsed_time})")

        # Train the model
        train_loss, train_accuracy = train_step(model, train_dataloader, loss_func, optimizer, device, scaler)

        scheduler.step()

        # Validate the model
        valid_loss, valid_accuracy = valid_step(model, valid_dataloader, loss_func, device)

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(model.state_dict(), "./checkpoints/dan_20M_best_model.pth")

        # Record the loss
        result["train_loss"].append(train_loss)
        result["valid_loss"].append(valid_loss)
        # Record the accuracy
        result["train_accuracy"].append(train_accuracy)
        result["valid_accuracy"].append(valid_accuracy)
        # Print the results for this epoch:
        if epoch:
            print(
                f"Epoch: {epoch}, train_loss: {train_loss}, train_accuracy: {train_accuracy}, "
                f"valid_loss: {valid_loss}, valid_accuracy: {valid_accuracy}"
            )
        elapsed_time = timedelta(seconds=(datetime.now() - start_time).seconds)

        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch}...")
                break

    # Return the results
    return result


class BoardDataSet(torch.utils.data.Dataset):
    def __init__(self, data_paths, target_paths):
        self.data_memmaps = [np.load(path, mmap_mode="r") for path in data_paths]
        self.target_memmaps = [np.load(path, mmap_mode="r") for path in target_paths]
        self.start_indices = [0] * len(data_paths)
        self.data_count = 0
        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        data = self.data_memmaps[memmap_index][index_in_memmap]
        target = self.target_memmaps[memmap_index][index_in_memmap]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


def load_data(train_size=0.8, batch_size=512):
    data_paths = []
    target_paths = []
    for i in range(1, 8):
        data_paths.append(f"./data/dan/20M/train_x_{i}.npz")
        target_paths.append(f"./data/dan/20M/train_y_{i}.npz")
    traindataset = BoardDataSet(data_paths, target_paths)
    print(traindataset.__len__())
    train_size = int(train_size * len(traindataset))
    valid_size = len(traindataset) - train_size
    train_dataset, valid_dataset = random_split(traindataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=20)
    return train_loader, valid_loader


def train_model(epochs=15):
    train_loader, valid_loader = load_data()
    model = Model().to(device)
    model.load_state_dict(torch.load("./checkpoints/dan_10M_best_model.pth"))
    loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start_time = timer()
    train(
        model,
        train_loader,
        valid_loader,
        loss_func,
        optimizer,
        scheduler,
        epochs,
        device,
    )
    end_time = timer()
    print(f"Total training time: {timedelta(seconds=end_time-start_time)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model()
