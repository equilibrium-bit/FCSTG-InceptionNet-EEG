import sys
sys.path.append('/tmp/electroencephalogram')
import torch
import numpy as np


def electrode_space_mapping(data, normalized_size):
    result = torch.zeros((data.shape[0], data.shape[1], normalized_size, normalized_size, normalized_size)).float().to(
        data.device)
    count = torch.zeros((normalized_size, normalized_size, normalized_size)).float().to(data.device)

    # 定义电极的三维坐标
    coordinates = [
        (-2.285379, 10.372299, 4.564709), (0.687462, 10.931931, 4.452579), (3.874373, 9.896583, 4.368097),
        (-2.82271, 9.895013, 6.833403), (4.143959, 9.607678, 7.067061), (-6.417786, 6.362997, 4.476012),
        (-5.745505, 7.282387, 6.764246), (-4.248579, 7.990933, 8.73188), (-2.046628, 8.049909, 10.162745),
        (0.716282, 7.836015, 10.88362), (3.193455, 7.889754, 10.312743), (5.337832, 7.691511, 8.678795),
        (6.842302, 6.643506, 6.300108), (7.197982, 5.671902, 4.245699), (-7.326021, 3.749974, 4.734323),
        (-6.882368, 4.211114, 7.939393), (-4.837038, 4.672796, 10.955297), (-2.677567, 4.478631, 12.365311),
        (0.455027, 4.186858, 13.104445), (3.654295, 4.254963, 12.205945), (5.863695, 4.275586, 10.714709),
        (7.610693, 3.851083, 7.604854), (7.821661, 3.18878, 4.400032), (-7.640498, 0.756314, 4.967095),
        (-7.230136, 0.725585, 8.331517), (-5.748005, 0.480691, 11.193904), (-3.009834, 0.621885, 13.441012),
        (0.341982, 0.449246, 13.839247), (3.62126, 0.31676, 13.082255), (6.418348, 0.200262, 11.178412),
        (7.743287, 0.254288, 8.143276), (8.214926, 0.533799, 4.980188), (-7.794727, -1.924366, 4.686678),
        (-7.103159, -2.735806, 7.908936), (-5.549734, -3.131109, 10.995642), (-3.111164, -3.281632, 12.904391),
        (-0.072857, -3.405421, 13.509398), (3.044321, -3.820854, 12.781214), (5.712892, -3.643826, 10.907982),
        (7.304755, -3.111501, 7.913397), (7.92715, -2.443219, 4.673271), (-7.161848, -4.799244, 4.411572),
        (-6.375708, -5.683398, 7.142764), (-5.117089, -6.324777, 9.046002), (-2.8246, -6.605847, 10.717917),
        (-0.19569, -6.696784, 11.505725), (2.396374, -7.077637, 10.585553), (4.802065, -6.824497, 8.991351),
        (6.172683, -6.209247, 7.028114), (7.187716, -4.954237, 4.477674), (-5.894369, -6.974203, 4.318362),
        (-5.037746, -7.566237, 6.585544), (-2.544662, -8.415612, 7.820205), (-0.339835, -8.716856, 8.249729),
        (2.201964, -8.66148, 7.796194), (4.491326, -8.16103, 6.387415), (5.766648, -7.498684, 4.546538),
        (-6.387065, -5.755497, 1.886141), (-3.542601, -8.904578, 4.214279), (-0.080624, -9.660508, 4.670766),
        (3.050584, -9.25965, 4.194428), (6.192229, -6.797348, 2.355135),
    ]

    # 将三维坐标转换为numpy数组
    coordinates_np = np.array(coordinates)
    # 获取X, Y, Z轴的最小值和最大值
    x_min, y_min, z_min = coordinates_np.min(axis=0)
    x_max, y_max, z_max = coordinates_np.max(axis=0)
    # 归一化并映射到整数坐标
    coordinates_normalized = (coordinates_np - [x_min, y_min, z_min]) / (
            coordinates_np.max(axis=0) - coordinates_np.min(axis=0)) * (normalized_size - 1)
    coordinates_int = np.rint(coordinates_normalized).astype(int)

    for i in range(len(coordinates_int)):
        x, y, z = coordinates_int[i]
        if count[x, y, z] == 0:
            result[:, :, x, y, z] = data[:, :, i]
        else:
            result[:, :, x, y, z] = (result[:, :, x, y, z] * count[x, y, z] + data[:, :, i]) / (count[x, y, z] + 1)
        count[x, y, z] += 1
    return result

def bipolar_electrode_space_mapping(data, normalized_size):
    result = torch.zeros((data.shape[0], data.shape[1], normalized_size, normalized_size, normalized_size)).float().to(
        data.device)
    count = torch.zeros((normalized_size, normalized_size, normalized_size)).float().to(data.device)
    coordinates = [(-4.3515825, 8.367647999999999, 4.5203605), (-7.029142, 3.5596555, 4.7215535), (-7.401173, -2.021465, 4.6893335), (-5.3522245, -6.851911, 4.3129255), (-3.266979, 9.181616, 6.6482945), (-4.998292, 4.235812, 9.962892), (-5.432547, -2.922043, 10.119952999999999), (-4.329845, -7.614677500000001, 6.6301404999999995), (4.6061024999999995, 8.794046999999999, 6.523446), (5.87809, 3.9458865000000003, 9.9286035), (5.6102065, -3.3121175, 10.0848815), (3.9263244999999998, -8.0420735, 6.5928895), (5.5361775, 7.7842424999999995, 4.306898), (7.706454, 3.1028505, 4.6129435), (7.701321, -2.210219, 4.728931), (5.11915, -7.1069435, 4.336051), (0.5291319999999999, 4.1426305, 12.3614335), (0.073146, -3.1237690000000002, 12.672486)]
    # 将三维坐标转换为numpy数组
    coordinates_np = np.array(coordinates)
    # 获取X, Y, Z轴的最小值和最大值
    x_min, y_min, z_min = coordinates_np.min(axis=0)
    x_max, y_max, z_max = coordinates_np.max(axis=0)
    # 归一化并映射到整数坐标
    coordinates_normalized = (coordinates_np - [x_min, y_min, z_min]) / (
            coordinates_np.max(axis=0) - coordinates_np.min(axis=0)) * (normalized_size - 1)
    coordinates_int = np.rint(coordinates_normalized).astype(int)

    for i in range(len(coordinates_int)):
        x, y, z = coordinates_int[i]
        if count[x, y, z] == 0:
            result[:, :, x, y, z] = data[:, :, i]
        else:
            result[:, :, x, y, z] = (result[:, :, x, y, z] * count[x, y, z] + data[:, :, i]) / (count[x, y, z] + 1)
        count[x, y, z] += 1
    return result


def electrode_space_mapping_2D(data):
    result = torch.zeros((data.shape[0],data.shape[1],9,9)).float().to(data.device)
    layer_out = [(0,3),(0,4),(0,5),(0,2),(0,6)]
    x,y = 1,0
    for i in range(45):
        layer_out.append((x,y))
        y += 1
        y %= 9
        if y == 0:
            x += 1
    layer_out.extend([(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(7,2),(7,3),(7,4),(7,5),(7,6)])
    for i in range(len(layer_out)):

        x,y = layer_out[i]
        result[:,:,x,y] = data[:,:,i]
    return result


def convert_bipolar_to_monopolar(tensor_data):
    """
    将双通道数据转换为单通道数据

    参数：
    tensor_data: torch.Tensor, 形状为 (batch_size, num_channels, time_length)
    bipolar_pairs: list, 包含双通道的对，e.g., [('FP1', 'F7'), ('F7', 'T7'), ...]
    reference_electrodes: dict, 包含参考电极及其电势，e.g., {'FP1': 0, 'O2': 0, 'CZ': 0}

    返回值：
    monopolar_tensor: torch.Tensor, 转换后的单通道数据，形状为 (batch_size, num_single_channels, time_length)
    """
    batch_size, num_channels, time_length = tensor_data.shape
    # 假设的双通道对
    bipolar_pairs = [
        ('FP1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'),
        ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('FP2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2'),
        ('FZ', 'CZ'), ('CZ', 'PZ')
    ]

    # 假设参考电极及其电势值
    reference_electrodes = {'FP1': torch.zeros(batch_size, time_length).to(tensor_data.device),
                            'O2': torch.zeros(batch_size, time_length).to(tensor_data.device),
                            'CZ': torch.zeros(batch_size, time_length).to(tensor_data.device)}

    # 初始化单通道电势字典
    monopolar_potentials = {elec: None for pair in bipolar_pairs for elec in pair}
    monopolar_potentials.update(reference_electrodes)

    # 迭代传播电势，直到所有电极的电势都被计算出来
    changed = True
    while changed:
        changed = False
        for i, (elec1, elec2) in enumerate(bipolar_pairs):
            if monopolar_potentials[elec1] is None and monopolar_potentials[elec2] is not None:
                monopolar_potentials[elec1] = tensor_data[:, i, :] + monopolar_potentials[elec2]
                changed = True
            elif monopolar_potentials[elec1] is not None and monopolar_potentials[elec2] is None:
                monopolar_potentials[elec2] = monopolar_potentials[elec1] - tensor_data[:, i, :]
                changed = True

    # 检查所有电极是否都有定义的电势
    for elec in monopolar_potentials:
        if monopolar_potentials[elec] is None:
            raise ValueError(f"电极 {elec} 的电势未定义，无法计算单通道电势")

    # 将结果转换为tensor
    monopolar_channels = sorted(monopolar_potentials.keys())
    monopolar_tensor = torch.stack([monopolar_potentials[elec] for elec in monopolar_channels], dim=1)

    return monopolar_tensor