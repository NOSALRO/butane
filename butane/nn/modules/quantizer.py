from typing import Tuple
import torch
import butane

class Quantizer(torch.nn.Module):
    def __init__(self, latent_dim: int, n_centers: int, device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._n_centers = n_centers
        self._device = device

        self.embedding = torch.nn.Embedding(self._n_centers, self._latent_dim)
        torch.nn.init.uniform_(self.embedding.weight, -1.0 / float(self._n_centers), 1.0 / float(self._n_centers))
        self.embedding.weight.requires_grad = True

        self._beta = 1.25
        self._reduction = 'mean'  # default reduction type

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Compute L2 distance between latents and embedding weights
            distance = torch.sum(x ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(x, self.embedding.weight.t())
            closest_encoding = torch.argmin(distance, dim=-1, keepdim=True).squeeze()

        # Quantized latents
        quantized_latents = self.embedding(closest_encoding)

        # Loss computation
        commitment_loss = (quantized_latents.detach() - x).pow(2)
        embedding_loss = (quantized_latents - x.detach()).pow(2)
        if self._reduction == 'sum':
           commitment_loss = commitment_loss.sum()
           embedding_loss  = embedding_loss.sum()
        elif self._reduction == 'mean':
           commitment_loss = commitment_loss.mean()
           embedding_loss  = embedding_loss.mean()
        quantization_loss = self._beta * commitment_loss + embedding_loss

        # Quantization with gradient copying trick
        quantized_latents = x + (quantized_latents - x).detach()

        return quantized_latents, quantization_loss

    def init_codebook(self, low: float, high: float) -> None:
        torch.nn.init.uniform_(self.embedding.weight, low, high)
        self.embedding.weight.requires_grad = True

    def init_codebook_kmeans(self, low: float, high: float, max_data: int = -1) -> None:
        max_data = self._n_centers * 400 if max_data == -1 else max_data
        rdata = torch.empty(max_data, self._latent_dim, device=self._device).uniform_(low, high)
        kmeans = butane.clustering.KMeans(n_centroids=self._n_centers, init='kmeans++')
        kmeans.fit(rdata)
        self.embedding.weight.data = kmeans.centroids
        self.embedding.weight.requires_grad = True

    def set_beta(self, new_beta: float) -> None:
        self._beta = new_beta

    def set_reduction(self, reduction: str) -> None:
        if reduction in ['mean', 'sum']:
            self._reduction = reduction
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'")

    def centers(self) -> torch.Tensor:
        return self.embedding.weight

    def beta(self) -> float:
        return self._beta


class Quantizer2d(Quantizer):

    def __init__(self, latent_dim: int, n_centers: int, device: torch.device = torch.device('cpu')) -> None:
        super().__init__(latent_dim, n_centers, device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)
        x_flat = x.reshape(B * H * W, C)

        with torch.no_grad():
            # Compute L2 distance between latents and embedding weights
            distance = torch.sum(x_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(x_flat, self.embedding.weight.t())
            closest_encoding = torch.argmin(distance, dim=-1, keepdim=True).squeeze()

        # Quantized latents
        quantized_latents = self.embedding(closest_encoding)
        quantized_latents = quantized_latents.view_as(x)

        # Loss computation
        commitment_loss = (quantized_latents.detach() - x).pow(2)
        embedding_loss = (quantized_latents - x.detach()).pow(2)
        if self._reduction == 'sum':
           commitment_loss = commitment_loss.sum()
           embedding_loss  = embedding_loss.sum()
        elif self._reduction == 'mean':
           commitment_loss = commitment_loss.mean()
           embedding_loss  = embedding_loss.mean()
        quantization_loss = self._beta * commitment_loss + embedding_loss

        # Quantization with gradient copying trick
        quantized_latents = x + (quantized_latents - x).detach()
        quantized_latents = quantized_latents.permute(0, 3, 1, 2)

        return quantized_latents, quantization_loss
