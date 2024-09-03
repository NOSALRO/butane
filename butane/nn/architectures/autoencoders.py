from typing import Optional, Callable, Tuple
import torch

class AE(torch.nn.Module):

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    @staticmethod
    def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(x_hat, x)

    def step(
        self,
        dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn_: Optional[Callable[[torch.Tensor, ...], Tuple[torch.Tensor,...]]] = loss_fn,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> None:
        sum_loss = 0.
        n_batches = len(dl)
        for n_batches, batch in enumerate(dl):
            if self.training:
                optimizer.zero_grad()
            x_reconstructed = self.forward(batch["data"])
            loss = loss_fn_(x_reconstructed, batch["data"])
            if self.training:
                loss.backward()
                optimizer.step()
            sum_loss += loss.item()

        if self.training and scheduler:
            scheduler.step()

        avg_loss = sum_loss / n_batches
        print(f"Loss: {avg_loss}")

class VQVAE(AE):

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, quantizer: torch.nn.Module) -> None:
        super().__init__(encoder, decoder)
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        q_z, quantization_loss = self.quantizer(z)
        reconstructed = self.decoder(q_z)
        return reconstructed, quantization_loss

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        q_z, _ = self.quantizer(z)
        return q_z

    def centers(self) -> torch.Tensor:
        return self.quantizer.centers()

    @staticmethod
    def loss_fn(x_hat, x, vq_loss) -> Tuple[torch.Tensor,...]:
        x_rec = torch.nn.functional.mse_loss(x_hat, x)
        loss = x_rec + vq_loss
        return loss, x_rec

    def step(
        self,
        dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn_: Optional[Callable[[torch.Tensor, ...], Tuple[torch.Tensor,...]]] = loss_fn,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> None:
        sum_loss = 0.
        sum_quantization_loss = 0.
        sum_loss_rec = 0.
        n_batches = len(dl)

        for batch in dl:
            if self.training:
                optimizer.zero_grad()
            x_reconstructed, quantization_loss = self.forward(batch["data"])
            loss, loss_rec = loss_fn_(x_reconstructed, batch["data"], quantization_loss)
            if (self.training):
                loss.backward()
                optimizer.step()

            sum_loss += loss.item()
            sum_quantization_loss += quantization_loss.item()
            sum_loss_rec += loss_rec.item()

        if self.training and scheduler:
            scheduler.step()

        avg_loss = sum_loss / n_batches
        avg_quantization_loss = sum_quantization_loss / n_batches
        avg_loss_rec = sum_loss_rec / n_batches
        print(f"Loss: {avg_loss} Reconstruction Loss: {avg_loss_rec} Quantization Loss: {avg_quantization_loss}")

class MLVQVAE(VQVAE):

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, quantizer: torch.nn.Module) -> None:
        super().__init__(encoder, decoder, quantizer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstructed = self.decoder(z)

        q_z, quantization_loss = self.quantizer(z)
        quantized_reconstructed = self.decoder(q_z)

        return reconstructed, quantized_reconstructed,  quantization_loss

    @staticmethod
    def loss_fn(x_hat, x_quantized_hat, x, vq_loss) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_rec = torch.nn.functional.mse_loss(x_hat, x)
        x_rec_quantized = torch.nn.functional.mse_loss(x_quantized_hat, x)
        loss = x_rec + x_rec_quantized + vq_loss
        return loss, x_rec, x_rec_quantized

    def step(
        self,
        dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn_: Optional[Callable[[torch.Tensor, ...], Tuple[torch.Tensor,...]]] = loss_fn,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ) -> None:
        sum_loss = 0.
        sum_quantization_loss = 0.
        sum_loss_ae = 0.
        sum_loss_vq = 0
        n_batches = len(dl)

        for batch in dl:
            if self.training:
                optimizer.zero_grad()
            x_reconstructed, x_quantized_reconstructed, quantization_loss = self.forward(batch["data"])
            loss, loss_ae, loss_vq = loss_fn_(x_reconstructed, x_quantized_reconstructed, batch["data"], quantization_loss)
            if (self.training):
                loss.backward()
                optimizer.step()

            sum_loss += loss.item()
            sum_quantization_loss += quantization_loss.item()
            sum_loss_ae += loss_ae.item()
            sum_loss_vq += loss_vq.item()

        if self.training and scheduler:
            scheduler.step()

        avg_loss = sum_loss / n_batches
        avg_quantization_loss = sum_quantization_loss / n_batches
        avg_loss_ae = sum_loss_ae / n_batches
        avg_loss_vq = sum_loss_vq / n_batches
        print(f"Loss: {avg_loss} Reconstruction Loss: {avg_loss_ae} Reconstruction Quant Loss: {avg_loss_vq} Quantization Loss: {avg_quantization_loss}")