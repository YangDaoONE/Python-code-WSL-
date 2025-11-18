import numpy as np


def conv2d_multi_in_multi_out(x, w, padding=0, stride=1):
    """
    x: 输入特征图，形状 (C_in, H_in, W_in)
    w: 卷积核，形状 (C_out, C_in, kH, kW)
    padding: 填充大小，整数，四周都填同样的圈数
    stride: 步幅，整数
    """
    C_in, H_in, W_in = x.shape
    C_out, C_in_w, kH, kW = w.shape

    assert C_in == C_in_w, "输入通道数必须和卷积核的输入通道数一致"

    # 1. 先做零填充
    if padding > 0:
        x_padded = np.pad(
            x,
            pad_width=((0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        x_padded = x

    _, H_pad, W_pad = x_padded.shape

    # 2. 计算输出特征图的高和宽
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    # 3. 准备输出张量
    out = np.zeros((C_out, H_out, W_out), dtype=float)

    # 4. 对每一个输出通道、每一个输出位置做卷积累加
    for co in range(C_out):                     # 遍历输出通道
        for i in range(H_out):                 # 遍历输出的每一行
            for j in range(W_out):             # 遍历输出的每一列
                # 当前卷积核在输入上的起始位置
                h_start = i * stride
                w_start = j * stride

                # 从填充后的输入中裁剪出对应的局部区域: 形状 (C_in, kH, kW)
                x_region = x_padded[:, 
                                      h_start:h_start + kH,
                                      w_start:w_start + kW]

                # 对所有输入通道做逐元素相乘然后求和
                out[co, i, j] = np.sum(x_region * w[co, :, :, :])

    return out


if __name__ == "__main__":

    in_channels = 3
    out_channels = 2
    H = W = 16
    kH = kW = 3
    padding = 1
    stride = 1

    # 随机生成 3 通道 16x16 的输入
    x = np.random.randn(in_channels, H, W)

    # 随机生成 2 个卷积核，每个卷积核有 3 个通道，大小 3x3
    w = np.random.randn(out_channels, in_channels, kH, kW)

    y = conv2d_multi_in_multi_out(x, w, padding=padding, stride=stride)
    print("输入形状:", x.shape)
    print("卷积核形状:", w.shape)
    print("输出形状:", y.shape)   