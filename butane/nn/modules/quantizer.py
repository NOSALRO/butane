from typing import Tuple
import torch
import butane

class AffineTransform(torch.nn.Module):
	def __init__(
			self,
			feature_size,
			use_running_statistics=False,
			momentum=0.1,
			lr_scale=1,
			num_groups=1,
			):
		super().__init__()

		self.use_running_statistics = use_running_statistics
		self.num_groups = num_groups

		if use_running_statistics:
			self.momentum = momentum
			self.register_buffer('running_statistics_initialized', torch.zeros(1))
			self.register_buffer('running_ze_mean', torch.zeros(num_groups, feature_size))
			self.register_buffer('running_ze_var', torch.ones(num_groups, feature_size))

			self.register_buffer('running_c_mean', torch.zeros(num_groups, feature_size))
			self.register_buffer('running_c_var', torch.ones(num_groups, feature_size))
		else:
			self.scale = torch.nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
			self.bias = torch.nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
			self.lr_scale = lr_scale
		return

	@torch.no_grad()
	def update_running_statistics(self, z_e, c):
		# we find it helpful to often to make an under-estimation on the
		# z_e embedding statistics. Empirically we observe a slight
		# over-estimation of the statistics, causing the straight-through
		# estimation to grow indefinitely. While this is not an issue
		# for most model architecture, some model architectures that don't
		# have normalized bottlenecks, can cause it to eventually explode.
        # placing the VQ layer in certain layers of ViT exhibits this behavior


		if self.training and self.use_running_statistics:
			unbiased = False

			ze_mean = z_e.mean([0, 1]).unsqueeze(0)
			ze_var = z_e.var([0, 1], unbiased=unbiased).unsqueeze(0)

			c_mean = c.mean([0]).unsqueeze(0)
			c_var = c.var([0], unbiased=unbiased).unsqueeze(0)

			if not self.running_statistics_initialized:
				self.running_ze_mean.data.copy_(ze_mean)
				self.running_ze_var.data.copy_(ze_var)
				self.running_c_mean.data.copy_(c_mean)
				self.running_c_var.data.copy_(c_var)
				self.running_statistics_initialized.fill_(1)
			else:
				self.running_ze_mean = (self.momentum * ze_mean) + (1 - self.momentum) * self.running_ze_mean
				self.running_ze_var = (self.momentum * ze_var) + (1 - self.momentum) * self.running_ze_var
				self.running_c_mean = (self.momentum * c_mean) + (1 - self.momentum) * self.running_c_mean
				self.running_c_var = (self.momentum * c_var) + (1 - self.momentum) * self.running_c_var

		# wd = 0.9998 # 0.995
		# self.running_ze_mean = wd * self.running_ze_mean
		# self.running_ze_var = wd * self.running_ze_var
		return


	def forward(self, codebook):
		scale, bias = self.get_affine_params()
		n, c = codebook.shape
		codebook = codebook.view(self.num_groups, -1, codebook.shape[-1])
		codebook = scale * codebook + bias
		return codebook.reshape(n, c)


	def get_affine_params(self):
		if self.use_running_statistics:
			scale = (self.running_ze_var / (self.running_c_var + 1e-8)).sqrt()
			bias = - scale * self.running_c_mean + self.running_ze_mean
		else:
			scale = (1. + self.lr_scale * self.scale)
			bias = self.lr_scale * self.bias
		return scale.unsqueeze(1), bias.unsqueeze(1)

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
        self.affine_transform = AffineTransform(
            self._latent_dim,
            use_running_statistics=False,
            lr_scale=2.,
            num_groups=1,
            )
        self.inplace_codebook_optimizer = torch.optim.AdamW(self.embedding.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute distances between inputs and embeddings

        self.affine_transform.update_running_statistics(x, self.embedding.weight)
        self.embedding.weight.data = self.affine_transform(self.embedding.weight)

        distance = torch.cdist(x, self.embedding.weight)
        closest_encoding = torch.argmin(distance, dim=-1, keepdim=True)

        # One-hot encoding of closest encodings
        encoding_one_hot = torch.zeros(closest_encoding.size(0), self._n_centers, device=x.device)
        encoding_one_hot.scatter_(1, closest_encoding, 1)

        # Quantized latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view_as(x)

        if self.training and hasattr(self, 'inplace_codebook_optimizer'):
			# update codebook inplace
            ((quantized_latents - x.detach()) ** 2).mean().backward()
            self.inplace_codebook_optimizer.step()
            self.inplace_codebook_optimizer.zero_grad()

			# forward pass again with the update codebook
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
        quantized_latents = x + (quantized_latents - x).detach() + (2. * quantized_latents) + (-2. * quantized_latents).detach()

        return quantized_latents, quantization_loss

    def init_codebook(self, low: float, high: float) -> None:
        torch.nn.init.uniform_(self.embedding.weight, low, high)
        self.embedding.weight.requires_grad = True

    def init_codebook_kmeans(self, low: float, high: float) -> None:
        rdata = torch.empty(self._n_centers * 400, self._latent_dim, device=self._device).uniform_(low, high)
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
