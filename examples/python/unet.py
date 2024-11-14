import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

if __name__ == "__main__":
    dev = torch.device('cuda')
    model = butane.nn.UNet1d([1, 100], [256], scale_factor=2, n_blocks=3)
    print(model(torch.randn(1, 1, 100)).size())

    emb = get_time_embedding(torch.linspace(0, 1, 100), 100)

    t_emb_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(100, 128)
            )
            for _ in range(2)
        ])

    x = t_emb_layers[0](emb)
    x = t_emb_layers[1](x)

