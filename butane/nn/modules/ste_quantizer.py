from typing import Tuple, Optional
import torch
from .quantizer import Quantizer

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

class STEQuantizer(Quantizer):
    def __init__(
        self,
        latent_dim: int,
        n_centers: int,
        sync_nu: Optional[float] = 0.0,
        affine_lr: Optional[float] = 0.0,
        affine_groups: Optional[int] = 1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__(latent_dim, n_centers, device)

        self._sync_nu = sync_nu
        self._affine_lr = affine_lr
        self._affine_groups = affine_groups
        self._has_optimizer = False

        if self._affine_lr > 0:
            self.affine_transform = AffineTransform(
                self._latent_dim,
                use_running_statistics=True,
                lr_scale=affine_lr,
                num_groups=1,
                )
        else:
            self.affine_transform = None

        if optimizer is not None:
            self.optimizer = optimizer(self.embedding.parameters())
            self._has_optimizer = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.affine_transform is not None:
            self.affine_transform.update_running_statistics(x, self.embedding.weight)
            self.embedding.weight.data = self.affine_transform(self.embedding.weight)
            self.embedding.weight.data = self.embedding.weight.data.to(x.device)

        with torch.no_grad():
            # Compute L2 distance between latents and embedding weights
            distance = torch.sum(x ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(x, self.embedding.weight.t())
            closest_encoding = torch.argmin(distance, dim=-1, keepdim=True).squeeze()

        # Quantized latents
        quantized_latents = self.embedding(closest_encoding)

        if self.training and self._has_optimizer:
            assert self._beta == 1.0, "Optimizer can only be used with beta=1.0"
            if self._reduction == 'sum':
                ((quantized_latents - x.detach()) ** 2).sum().backward()
            elif self._reduction == 'mean':
                ((quantized_latents - x.detach()) ** 2).mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # forward pass again with the update codebook
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
        if self._sync_nu > 0.:
            quantized_latents = x + (quantized_latents - x).detach() + (self._sync_nu * quantized_latents) + (-self._sync_nu * quantized_latents).detach()
        else:
            quantized_latents = x + (quantized_latents - x).detach()

        return quantized_latents, quantization_loss

    def set_sync_nu(self, new_sync_nu: float) -> None:
        self._sync_nu = new_sync_nu

    def sync_nu(self) -> float:
        return self._sync_nu

    def set_affine_lr(self, new_affine_lr: float) -> None:
        self._affine_lr = new_affine_lr

    def affine_lr(self) -> float:
        return self._affine_lr


class STEQuantizer2d(STEQuantizer):
    def __init__(
        self,
        latent_dim: int,
        n_centers: int,
        sync_nu: Optional[float] = 0.0,
        affine_lr: Optional[float] = 0.0,
        affine_groups: Optional[int] = 1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super().__init__(latent_dim, n_centers, sync_nu, affine_lr, affine_groups, optimizer, device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)
        x_flat = x.reshape(B * H * W, C)

        if self.affine_transform is not None:
            self.affine_transform.update_running_statistics(x, self.embedding.weight)
            self.embedding.weight.data = self.affine_transform(self.embedding.weight)
            self.embedding.weight.data = self.embedding.weight.data.to(x.device)

        with torch.no_grad():
            # Compute L2 distance between latents and embedding weights
            distance = torch.sum(x_flat ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - \
                2 * torch.matmul(x_flat, self.embedding.weight.t())
            closest_encoding = torch.argmin(distance, dim=-1, keepdim=True).squeeze()

        # Quantized latents
        quantized_latents = self.embedding(closest_encoding)
        quantized_latents = quantized_latents.view_as(x)

        if self.training and self._has_optimizer:
            assert self._beta == 1.0, "Optimizer can only be used with beta=1.0"
            if self._reduction == 'sum':
                ((quantized_latents - x.detach()) ** 2).sum().backward()
            elif self._reduction == 'mean':
                ((quantized_latents - x.detach()) ** 2).mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # forward pass again with the update codebook
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
        if self._sync_nu > 0.:
            quantized_latents = x + (quantized_latents - x).detach() + (self._sync_nu * quantized_latents) + (-self._sync_nu * quantized_latents).detach()
        else:
            quantized_latents = x + (quantized_latents - x).detach()

        quantized_latents = quantized_latents.permute(0, 3, 1, 2)
        return quantized_latents, quantization_loss
