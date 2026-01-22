import torch
from torch import nn
import torch.nn.functional as F


class Inception3DModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception3DModule, self).__init__()
        self.branch3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch7x7 = nn.Conv3d(in_channels, out_channels, kernel_size=7, padding=3)

        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch3x3 = F.relu(self.branch3x3(x))
        branch5x5 = F.relu(self.branch5x5(x))
        branch7x7 = F.relu(self.branch7x7(x))
        branch_pool = F.relu(self.branch_pool(x))
        branch_pool = F.interpolate(branch_pool, size=(x.size(2), x.size(3), x.size(4)), mode='trilinear', align_corners=False)

        outputs = [branch3x3, branch5x5, branch7x7, branch_pool]
        return torch.cat(outputs, 1)


class Inception3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super(Inception3DCNN, self).__init__()
        self.inception1 = Inception3DModule(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.inception2 = Inception3DModule(128, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.inception3 = Inception3DModule(256, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Adjust the input size for the fully connected layer
        fc_input_size = 512 * (input_size // 8) * (input_size // 8) * (input_size // 8)
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.inception1(x))
        x = self.pool2(self.inception2(x))
        x = self.pool3(self.inception3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch3x3 = F.relu(self.branch3x3(x))
        branch5x5 = F.relu(self.branch5x5(x))
        branch7x7 = F.relu(self.branch7x7(x))
        branch_pool = F.relu(self.branch_pool(x))
        branch_pool = F.interpolate(branch_pool, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        outputs = [branch3x3, branch5x5, branch7x7, branch_pool]
        return torch.cat(outputs, 1)


class Inception2DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super(Inception2DCNN, self).__init__()
        self.inception1 = InceptionModule(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.inception2 = InceptionModule(128, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.inception3 = InceptionModule(256, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Adjust the input size for the fully connected layer
        fc_input_size = 512 * (input_size // 8) * (input_size // 8)
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.inception1(x))
        x = self.pool2(self.inception2(x))
        x = self.pool3(self.inception3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Lightweight2DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super(Lightweight2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Adjust the input size for the fully connected layer
        fc_input_size = 128 * (input_size // 8) * (input_size // 8)
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Lightweight3DCNN(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super(Lightweight3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # 调整全连接层的输入尺寸
        fc_input_size = 128 * (input_size // 8) * (input_size // 8) * (input_size // 8)
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x