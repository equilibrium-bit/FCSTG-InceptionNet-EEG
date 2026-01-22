import sys
sys.path.append('/tmp/electroencephalogram')
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv3d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, depth * height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, depth, height, width)
        out = self.gamma * out + x
        return out

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.self_attention = SelfAttention(in_dim)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.self_attention(x)
        x = self.global_avg_pool(x)
        return x


class MS_CAM(nn.Module):
    def __init__(self, channels, map_size,reduction_ratio=4):
        super(MS_CAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        # Global context1
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels // reduction_ratio)
        self.fc2 = nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)



        # Multi-scale pooling1
        self.multi_scale_pool1 = nn.AvgPool3d(kernel_size=((map_size+1)//2,(map_size+1)//2,(map_size+1)//2),stride=(map_size//2,map_size//2,map_size//2))
        self.multi_fc11 = nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False)
        self.multi_bn11= nn.BatchNorm3d(channels // reduction_ratio)
        self.multi_fc12 = nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        self.multi_bn12 = nn.BatchNorm3d(channels)

        # Global context2
        self.attention_pooling1 = AttentionPooling(channels)

        # Multi-scale pooling2
        self.multi_scale_pool2 = nn.AvgPool3d(
            kernel_size=((map_size + 1) // 2, map_size, map_size),
            stride=(map_size // 2, map_size, map_size))
        self.multi_fc21 = nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False)
        self.multi_bn21 = nn.BatchNorm3d(channels // reduction_ratio)
        self.multi_fc22 = nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        self.multi_bn22 = nn.BatchNorm3d(channels)

        # Global context3
        self.attention_pooling2 = AttentionPooling(channels)

        # Local context
        self.local_conv1 = nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels // reduction_ratio)
        self.local_conv2 = nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(channels)

        # Weights for multi-scale features
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight4 = nn.Parameter(torch.ones(1))

        self.channel_alignment = nn.Conv3d(channels*3,channels,1)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()

        # Global context
        global_context = self.global_avg_pool(x)
        global_context = self.fc1(global_context)
        global_context = self.bn1(global_context)
        global_context = F.relu(global_context)
        global_context = self.fc2(global_context)
        global_context = self.bn2(global_context)
        # global_context = global_context.expand_as(x)

        # Multi-scale pooling1
        context1 = self.multi_scale_pool1(x)
        context1 = self.multi_fc11(context1)
        context1 = self.multi_bn11(context1)
        context1 = F.relu(context1)
        context1 = self.multi_fc12(context1)
        context1 = self.multi_bn12(context1)
        global_context1 = self.attention_pooling1(context1)
        context1 = F.interpolate(context1, size=(d, h, w), mode='nearest')

        # Multi-scale pooling2
        context2 = self.multi_scale_pool2(x)
        context2 = self.multi_fc21(context2)
        context2 = self.multi_bn21(context2)
        context2 = F.relu(context2)
        context2 = self.multi_fc22(context2)
        context2 = self.multi_bn22(context2)
        global_context2 = self.attention_pooling2(context2)
        context2 = F.interpolate(context2, size=(d, h, w), mode='nearest')

        # Local context
        local_context = self.local_conv1(x)
        local_context = self.bn3(local_context)
        local_context = F.relu(local_context)
        local_context = self.local_conv2(local_context)
        local_context = self.bn4(local_context)

        global_context = torch.cat([global_context,global_context1,global_context2],dim=1)
        global_context = self.channel_alignment(global_context)
        global_context = global_context.expand_as(x)
        
        # Combining global, local, and multi-scale contexts with additional transformation
        combined_context = self.weight1 * global_context + self.weight2 * local_context + self.weight3 * context1 + self.weight4 * context2

        # Attention weights
        attention = self.sigmoid(combined_context)

        return attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(2, 1, 5, padding=2, bias=False)
        self.conv3 = nn.Conv3d(2, 1, 7, padding=3, bias=False)
        self.conv = nn.Conv3d(3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        energy1 = self.conv1(x)
        energy2 = self.conv2(x)
        energy3 = self.conv3(x)
        energy = torch.cat([energy1,energy2,energy3],dim=1)
        energy = self.conv(energy)
        attention =  self.sigmoid(energy)
        return attention

class OriginalSpatialAttention(nn.Module):
    def __init__(self):
        super(OriginalSpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attention =  self.sigmoid(self.conv(x))
        return attention

class PathSelectionAttention(nn.Module):
    def __init__(self, channels, map_size,initial_temperature=5.0, min_temperature=0.5, anneal_rate=0.99):
        super(PathSelectionAttention, self).__init__()
        self.ms_cam = MS_CAM(channels,map_size)
        self.spatial_attention = SpatialAttention()
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

    def gumbel_softmax_sample(self, logits, temperature):
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def forward(self, x):
        # Path 1: Channel first, then spatial
        ms_attention_1 = self.ms_cam(x)
        sa_attention_1 = self.spatial_attention(ms_attention_1 * x)

        # Path 2: Spatial first, then channel
        sa_attention_2 = self.spatial_attention(x)
        ms_attention_2 = self.ms_cam(sa_attention_2 * x)

        # Path 3: Channel and spatial independently, then add
        ms_attention_3 = self.ms_cam(x)
        sa_attention_3 = self.spatial_attention(x)
        combined_add = ms_attention_3 + sa_attention_3
        combined_add = F.softmax(combined_add, dim=1)  # 对权重重新进行softmax

        # Path 4: Channel and spatial independently, then multiply
        combined_mul = ms_attention_3 * sa_attention_3
        combined_mul = F.softmax(combined_mul, dim=1)  # 对权重重新进行softmax
        # Expand dimensions to match paths 3 and 4
        sa_attention_1_expanded = sa_attention_1.expand_as(combined_add)
        ms_attention_2_expanded = ms_attention_2.expand_as(combined_add)
        # Stack all paths
        paths = torch.stack([sa_attention_1_expanded, ms_attention_2_expanded, combined_add, combined_mul], dim=-1)

        # Gumbel Softmax to select one path
        logits = torch.ones(paths.size(-1))  # equal logits for each path
        weights = self.gumbel_softmax_sample(logits, self.temperature).to(paths.device)
        # Select one path based on weights
        selected_path = torch.sum(paths * weights.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1)
        # 更新温度
        self.temperature = max(self.temperature * self.anneal_rate, self.min_temperature)
        return selected_path


class IterativeAttentionFeatureFusion(nn.Module):
    def __init__(self, channels,times,map_size = 9):
        super(IterativeAttentionFeatureFusion, self).__init__()
        self.channels = channels
        self.times = times
        self.path_selection_attention = PathSelectionAttention(channels,map_size)
        # self.multi_channel_attention = MS_CAM(channels,map_size)

    def forward(self, x, y):
        combined = x + y
        attention_weights = self.path_selection_attention(combined)
        fused_features = attention_weights * x + (1 - attention_weights) * y
        for i in range(self.times-1):
            attention_weights = self.path_selection_attention(fused_features)
            fused_features = attention_weights * x + (1 - attention_weights) * y
        return fused_features


class MS_CAM_3D(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM_3D, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return wei