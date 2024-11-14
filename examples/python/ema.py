import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dev = torch.device('cuda')
    ds = butane.data.Dataset(torch.jit.load("data/mnist_data.pt").state_dict()['0'])
    ds.to(dev)
    butane.data.ops.drop_to_max_size(ds, 8000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    c_enc = butane.nn.Conv2dBlock(
        input_dims = [1, 28, 28],
        channels = [32, 32],
        activation_function = [torch.nn.GELU()],
        conv_stride = [1, 1],
        conv_bias = [True, True],
        pool_kernels = [0, 0],
        normalization = [True, False]
    )

    mlp_enc = butane.nn.MLPBlock(
        input_dims = c_enc.output_size.prod().item(),
        output_dims=2,
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
        output_activation=False)

    mlp_dec = butane.nn.MLPBlock(
        input_dims = 2,
        output_dims=c_enc.output_size.prod().item(),
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()],
        output_activation=False)

    c_dec = butane.nn.ConvTranspose2dBlock(
        input_dims = [c_enc.output_size[0].item(), c_enc.output_size[1].item(), c_enc.output_size[2].item()],
        channels = [32, 1],
        activation_function = [torch.nn.GELU(), torch.nn.Sigmoid()],
        conv_stride = [1, 1],
        conv_bias = [True, True],
        output_activation = False,
        normalization = [False, False])

    quantizer = butane.nn.STEQuantizer(2, 100, affine_lr=1.5, sync_nu=2.0, optimizer=torch.optim.Adam, device=dev)
    quantizer.set_beta(1.)
    quantizer.init_codebook_kmeans(-1./100., 1./100.)
    quantizer.affine_transform.set_running_statistics(True)
    quantizer.affine_transform.set_num_groups(2)

    encoder = torch.nn.Sequential(c_enc, torch.nn.Flatten(1), mlp_enc)
    decoder = torch.nn.Sequential(mlp_dec, butane.nn.Unflatten(1, c_enc.output_size), c_dec)
    model = butane.nn.MLVQVAE(encoder, decoder, quantizer).to(dev)
    print(model)

    ema = butane.nn.EMA(model, 0.99)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-03, weight_decay=0.01)
    trainer = butane.nn.utils.ModelTrainer(model, dl, optimizer)
    for i in range(20):
        sum_loss = 0.
        sum_quantization_loss = 0.
        sum_loss_ae = 0.
        sum_loss_vq = 0
        n_batches = len(dl)

        for batch in dl:
            optimizer.zero_grad()
            x_reconstructed, x_quantized_reconstructed, quantization_loss = model(batch["data"])
            loss, loss_ae, loss_vq = model.loss_fn(x_reconstructed, x_quantized_reconstructed, batch["data"], quantization_loss)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_quantization_loss += quantization_loss.item()
            sum_loss_ae += loss_ae.item()
            sum_loss_vq += loss_vq.item()
            ema.update()

        avg_loss = sum_loss / n_batches
        avg_quantization_loss = sum_quantization_loss / n_batches
        avg_loss_ae = sum_loss_ae / n_batches
        avg_loss_vq = sum_loss_vq / n_batches
        print(f"Loss: {avg_loss} Reconstruction Loss: {avg_loss_ae} Reconstruction Quant Loss: {avg_loss_vq} Quantization Loss: {avg_quantization_loss}")

    ema.apply_ema()
    model.eval()

    fig, ax = plt.subplots()
    latents = model.encode(ds.data).detach().cpu().numpy()
    centers = model.centers().detach().cpu().numpy()
    ax.scatter(latents[:, 0], latents[:, 1])
    ax.scatter(centers[:, 0], centers[:, 1])
    plt.show()
    for d in ds.data.clone():
        fig, ax = plt.subplots(1,3)
        dn = d.squeeze().cpu().numpy()
        drq, dr, _ = model(d.unsqueeze(0))
        drn = dr.squeeze().detach().cpu().numpy()
        drqn = drq.squeeze().detach().cpu().numpy()
        ax[0].imshow(dn)
        ax[1].imshow(drn)
        ax[2].imshow(drqn)
        plt.show()
