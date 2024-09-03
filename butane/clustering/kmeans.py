from typing import Optional
import torch

class KMeans:

    def __init__(
        self,
        n_centroids: int,
        init: Optional[str] = 'kmeans++',
        random_state: Optional[int] = -1,
        tol: Optional[float] = 1e-8
    ) -> None:
        self.n_centroids = n_centroids
        self.init = init
        self.tol = tol
        self.centroids = None
        torch.random.manual_seed(random_state)

    def fit(self, x: torch.Tensor) -> None:
        if self.init == 'kmeans++':
            self.__plusplus(x)
        else:
            self.centroids = torch.distributions.Uniform(low=x.min(), high=x.max()).rsample((self.n_centroids, x.size(-1)))
            self.centroids = self.centroids.to(x.device)

        prev_centroids = torch.full_like(self.centroids, torch.finfo(self.centroids.dtype).max)
        while torch.linalg.norm(self.centroids - prev_centroids) > self.tol:
            dist = torch.cdist(x, self.centroids)
            closest = dist.argmin(-1)
            prev_centroids = self.centroids
            mask = closest == torch.arange(self.n_centroids, device=x.device).unsqueeze(1)
            for j in torch.arange(self.n_centroids):
                self.centroids[j] = x[closest == j].mean(0)

    def __plusplus(self, x: torch.Tensor) -> None:
        centroid_idx = torch.randint(len(x), size=(1,), dtype=torch.int32)
        self.centroids = x[centroid_idx]

        while self.centroids.size(0) != self.n_centroids:
            dist = torch.cdist(x, self.centroids).min(-1).values
            centroid_idx = torch.multinomial(dist.pow(2), 1)
            self.centroids = torch.vstack([self.centroids, x[centroid_idx]])