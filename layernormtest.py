import torch
import torch.nn as nn

def layer_norm_process(feature: torch.Tensor, beta=0., gamma=1., eps=1e-5):
    var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
    # 均值
    mean = var_mean[1]
    # 方差
    var = var_mean[0]

    # layer norm process
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature


def main():
    # t = torch.rand(4, 2, 3)
    t = torch.tensor(
        [
            [[0.2647, 0.6757, 0.4979],
            [0.7594, 0.5378, 0.7707]],
            
            [[0.7527, 0.6625, 0.57],
            [0.9368, 0.3376, 0.7222]],
            
            [[0.2775, 0.3694, 0.6515],
            [0.4018, 0.0814, 0.0907]],
            
            [[0.5589, 0.7082, 0.3384],
            [0.3242, 0.4460, 0.5619]]
        ]
    )
    print(t)
    # 仅在最后一个维度上做norm处理
    norm = nn.LayerNorm(normalized_shape=t.shape[-1], eps=1e-5)
    # 官方layer norm处理
    t1 = norm(t)
    # 自己实现的layer norm处理
    t2 = layer_norm_process(t, eps=1e-5)
    print("t1:\n", t1)
    print("t2:\n", t2)


if __name__ == '__main__':
    main()
