import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dev = torch.device("cuda")
    ds = butane.data.Dataset(torch.jit.load("data/mnist_data.pt").state_dict()["0"])
    ds.to(dev)
    test_ds = ds.split(0.9)
    butane.data.ops.drop_to_max_size(ds, 8000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    model = butane.nn.UNet2d(
        [1, 28, 28],
        32,
        channel_mults=(1, 2, 4),
        self_condition=True,
        attention=True,
        use_film=True,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-03, weight_decay=0.01)

    epochs = 5
    for epoch in range(epochs):
        sum_loss = 0
        for batch in dl:
            optimizer.zero_grad()
            pred = model(batch["data"])
            loss = torch.nn.functional.mse_loss(pred, batch["data"])
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        print(f"Epochs {epoch} -> Loss: {sum_loss/len(dl)}")

    model.eval()

    with torch.no_grad():
        test_input = test_ds[:100]["data"]
        reconstructed = model(test_input).squeeze().cpu()
        test_input = test_input.squeeze().cpu()
        for i in range(reconstructed.size(0)):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(reconstructed[i])
            ax[1].imshow(test_input[i])
            plt.show()
