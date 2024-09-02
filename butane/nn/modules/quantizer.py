import torch
# from torch_kmeans import KMeans

class Quantizer(torch.nn.Module):
    def __init__(self, latent_dim, n_centers, device=torch.device('cpu')):
        super(Quantizer, self).__init__()
        self._latent_dim = latent_dim
        self._n_centers = n_centers
        self._device = device

        self.embedding = torch.nn.Embedding(self._n_centers, self._latent_dim)
        torch.nn.init.uniform_(self.embedding.weight, -1.0 / float(self._n_centers), 1.0 / float(self._n_centers))
        self.embedding.weight.requires_grad = True

        self._beta = 1.25
        self._reduction = 'mean'  # default reduction type

    def forward(self, x):
        # Compute distances between inputs and embeddings
        distance = torch.cdist(x, self.embedding.weight)
        closest_encoding = torch.argmin(distance, dim=-1, keepdim=True)

        # One-hot encoding of closest encodings
        encoding_one_hot = torch.zeros(closest_encoding.size(0), self._n_centers, device=x.device)
        encoding_one_hot.scatter_(1, closest_encoding, 1)

        # Quantized latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
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

        return quantized_latents, quantization_loss

    def init_codebook(self, low, high):
        torch.nn.init.uniform_(self.embedding.weight, low, high)
        self.embedding.weight.requires_grad = True

    # def init_codebook_kmeans(self, low, high):
    #     rdata = torch.empty(self._n_centers * 400, self._latent_dim, device=self._device).uniform_(low, high)
    #     kmeans = KMeans(n_clusters=self._n_centers, mode='euclidean', verbose=0)
    #     kmeans.fit_predict(rdata)
    #     self.embedding.weight.data = kmeans.centroids
    #     self.embedding.weight.requires_grad = True

    def set_beta(self, new_beta):
        self._beta = new_beta

    def set_reduction(self, reduction):
        if reduction in ['mean', 'sum']:
            self._reduction = reduction
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'")

    def centers(self):
        return self.embedding.weight

    def beta(self):
        return self._beta


class Quantizer2d(Quantizer):
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Compute distances between inputs and embeddings
        distance = torch.cdist(x, self.embedding.weight)
        closest_encoding = torch.argmin(distance, dim=-1, keepdim=True)

        # One-hot encoding of closest encodings
        encoding_one_hot = torch.zeros(closest_encoding.size(0), self._n_centers, device=x.device)
        encoding_one_hot.scatter_(1, closest_encoding, 1)

        # Quantized latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view_as(x)

        # Loss computation
        commitment_loss = torch.nn.functional.mse_loss(quantized_latents.detach(), x, reduction=self._reduction)
        embedding_loss = torch.nn.functional.mse_loss(quantized_latents, x.detach(), reduction=self._reduction)
        quantization_loss = self._beta * commitment_loss + embedding_loss

        # Quantization with gradient copying trick
        quantized_latents = x + (quantized_latents - x).detach()
        quantized_latents = quantized_latents.view(B, H, W, C).permute(0, 3, 1, 2)

        return quantized_latents, quantization_loss
