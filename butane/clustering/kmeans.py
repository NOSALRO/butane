from typing import Optional
import torch

class KMeans:

    def __init__(
        self,
        n_centroids: int,
        init: Optional[str] = 'kmeans++',
        max_iters: Optional[int] = 500,
        tol: Optional[float] = 1e-4,
        random_state: Optional[int] = -1,
    ) -> None:
        self.n_centroids = int(n_centroids)
        self.init = init
        self.tol = tol
        self.centroids = None
        self._max_iters = max_iters
        if random_state != -1:
            torch.random.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)


    def fit(self, x: torch.Tensor) -> None:
        if self.init == 'kmeans++':
            self._init_plusplus(x)
        elif self.init == 'random':
            self._init_random(x)
        else:
            raise ValueError('KMeans init method does not exits')

        for _ in range(self._max_iters):
            prev_centroids = self.centroids.clone()

            dist = torch.cdist(x, self.centroids)
            closest = dist.argmin(-1)

            assignments_per_centroid_count = torch.bincount(closest, minlength=self.n_centroids)
            sum_of_x_per_centroid = torch.zeros_like(self.centroids).scatter_add_(0, closest.unsqueeze(1).expand_as(x), x)

            valid_centroids = assignments_per_centroid_count > 0
            self.centroids[valid_centroids] = sum_of_x_per_centroid[valid_centroids] / assignments_per_centroid_count[valid_centroids].unsqueeze(1)
            if torch.linalg.norm(self.centroids - prev_centroids) < self.tol:
                break

    def _init_random(self, x: torch.Tensor) -> None:
        self.centroids = torch.distributions.Uniform(low=x.min(), high=x.max()).rsample((self.n_centroids, x.size(-1)))
        self.centroids = self.centroids.to(x.device)

    def _init_plusplus(self, x: torch.Tensor) -> None:
        centroid_idx = torch.randint(len(x), size=(1,), dtype=torch.int32)
        track_selected = torch.zeros(x.size(0)).bool()
        track_selected[centroid_idx] = True
        distances = torch.zeros((x.size(0), self.n_centroids), device=x.device)

        for i in range(self.n_centroids - 1):
            distances[:, i] = torch.cdist(x, x[centroid_idx]).pow(2).squeeze()
            centroid_idx = torch.multinomial(distances[:, :i+1].min(-1).values, 1)
            track_selected[centroid_idx] = True
        self.centroids = x[track_selected]

class MiniBatchKMeans(KMeans):

    def __init__(
        self,
        n_centroids: int,
        init: Optional[str] = 'kmeans++',
        batch_size: Optional[int] = 1024,
        max_iters: Optional[int] = 500,
        tol: Optional[float] = 1e-4,
        random_state: Optional[int] = -1,
    ) -> None:
        super().__init__(n_centroids, init, max_iters, tol, random_state)
        self._batch_size = batch_size

    def fit(self, x: torch.Tensor) -> None:
        if self.init == 'kmeans++':
            self._init_plusplus(x)
        elif self.init == 'random':
            self._init_random(x)
        else:
            raise ValueError('Invalid KMeans init method!')

        for i in range(self._max_iters):
            prev_centroids = self.centroids.clone()

            mini_batch_idx = torch.randint(0, x.size(0), (self._batch_size,), device=x.device)
            mini_batch = x[mini_batch_idx]

            dist = torch.cdist(mini_batch, self.centroids)
            closest = dist.argmin(-1)

            assignments_per_centroid_count = torch.bincount(closest, minlength=self.n_centroids)

            sum_of_x_per_centroid = torch.zeros_like(self.centroids).scatter_add_(0, closest.unsqueeze(1).expand_as(mini_batch), mini_batch)

            valid_centroids = assignments_per_centroid_count > 0
            eta = (1./assignments_per_centroid_count[valid_centroids]).unsqueeze(1)
            self.centroids[valid_centroids] = (
                (1 - eta) * self.centroids[valid_centroids] +
                eta * sum_of_x_per_centroid[valid_centroids] / assignments_per_centroid_count[valid_centroids].unsqueeze(1)
            )

            if torch.linalg.norm(self.centroids - prev_centroids) < self.tol:
                break