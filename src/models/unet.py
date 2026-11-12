import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Блок энкодера: две свёртки + остаточное соединение + уменьшение размера вдвое."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_a = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv_b = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(out_ch)
        self.bn_b = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d = self.conv_a(x)                       # результат первой свёртки запоминаем
        y = self.act(self.bn_a(d))
        y = self.act(self.bn_b(self.conv_b(y)))
        y = y + d                                # остаточное соединение: добавляем запомненное
        return y, self.pool(y)                   # y идёт в декодер (skip), pool(y) на следующий уровень


class BottleNeck(nn.Module):
    """Самая глубокая часть: две свёртки, размер не меняется."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Блок декодера: увеличить вдвое, склеить с skip из энкодера, две свёртки."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_a = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.conv_b = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(out_ch)
        self.bn_b = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, skip):
        x = torch.cat([self.up(x), skip], dim=1)   # склейка по оси каналов (dim=1 в PyTorch)
        x = self.act(self.bn_a(self.conv_a(x)))
        return self.act(self.bn_b(self.conv_b(x)))


class UNet(nn.Module):
    def __init__(self, f=(16, 32, 64, 128, 256)):
        super().__init__()
        self.down1 = DownBlock(3, f[0])
        self.down2 = DownBlock(f[0], f[1])
        self.down3 = DownBlock(f[1], f[2])
        self.down4 = DownBlock(f[2], f[3])
        self.bottleneck = BottleNeck(f[3], f[4])
        self.up1 = UpBlock(f[4], f[3], f[3])
        self.up2 = UpBlock(f[3], f[2], f[2])
        self.up3 = UpBlock(f[2], f[1], f[1])
        self.up4 = UpBlock(f[1], f[0], f[0])
        self.head = nn.Conv2d(f[0], 1, 1)            # 1x1 свёртка: 16 каналов -> 1 карта глубины

    def forward(self, x):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)
        b = self.bottleneck(p4)
        u = self.up1(b, c4)
        u = self.up2(u, c3)
        u = self.up3(u, c2)
        u = self.up4(u, c1)
        return self.head(u)                          # линейный выход, без активации