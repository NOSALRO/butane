import torch
import numpy as np
import butane

if __name__ == "__main__":
    dev = torch.device('cuda')
    ds = butane.data.Dataset(torch.jit.load("data/mnist_data.pt").state_dict()['0'])
    ds.to(dev)
    butane.data.ops.drop_to_max_size(ds, 3000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    c_enc = butane.nn.Conv2dBlock(
        input_dims = [1, 28, 28],
        channels = [32, 32],
        activation_function = [torch.nn.GELU(), torch.nn.GELU()],
        conv_stride = [[1], [1]],
        conv_bias = [True, True],
        pool_kernels = [[0], [0]],
        normalization = [True, True]
    )

    mlp_enc = butane.nn.MLPBlock(
        input_dims = c_enc.output_size.prod().item(),
        output_dims=100,
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
        output_activation=False)

    mlp_dec = butane.nn.MLPBlock(
        input_dims = 100,
        output_dims=c_enc.output_size.prod().item(),
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()],
        output_activation=False)

    c_dec = butane.nn.ConvTranspose2dBlock(
        input_dims = [32, 28, 28],
        channels = [32, 1],
        activation_function = [torch.nn.GELU(), torch.nn.Sigmoid()],
        conv_stride = [[1], [1]],
        conv_bias = [True, True],
        pool_kernels = [[0], [0]],
        output_activation = False,
        normalization = [False, False])

    quantizer = butane.nn.Quantizer (100, 10, dev)
    quantizer.set_beta(1.25)
    quantizer.init_codebook_kmeans(-1., 1.)

    encoder = torch.nn.Sequential(c_enc, torch.nn.Flatten(1), mlp_enc)
    decoder = torch.nn.Sequential(mlp_dec, butane.nn.Unflatten(1, c_enc.output_size[1:]), c_dec)
    model = butane.nn.MLVQVAE(encoder, decoder, quantizer).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-03)
    trainer = butane.nn.utils.ModelTrainer(model, dl, optimizer)
    trainer(100, model.loss_fn)
