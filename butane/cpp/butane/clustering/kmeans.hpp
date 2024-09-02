#pragma once

#include "../types.hpp"
#include <limits>
#include <torch/torch.h>

namespace butane {
    namespace clustring {
        class KMeans {
        public:
            KMeans(int n_centroids, KMeansInit init = KMeansPlusPlus, double tol = 1e-08, int random_state = -1) : _n_centroids(n_centroids), _init(init), _tol(tol)
            {
                random_state == -1 ? torch::cuda::manual_seed(time(0)) : torch::cuda::manual_seed(random_state);
                random_state == -1 ? torch::manual_seed(time(0)) : torch::manual_seed(random_state);
            };

            void fit(const torch::Tensor& x)
            {
                _centroids = torch::empty({_n_centroids, x.size(1)}).to(x.device());
                switch (_init) {
                case KMeansPlusPlus:
                    this->_plusplus(x);

                case Random:
                    torch::Tensor high_, low_;
                    high_ = std::get<0>(torch::max(x, 0)), low_ = std::get<0>(torch::min(x, 0));
                    for (int i = 0; i < x.sizes()[1]; ++i) {
                        _centroids.index({torch::indexing::Slice(), i}).uniform_(low_[i].item<double>(), high_[i].item<double>());
                    }
                }

                torch::Tensor _prev_centroids = torch::full_like(_centroids, std::numeric_limits<float>::max());
                while (torch::linalg_norm(_centroids - _prev_centroids).item<double>() > _tol) {
                    torch::Tensor dist = torch::cdist(x, _centroids);
                    torch::Tensor closest = dist.argmin(-1);
                    _prev_centroids = _centroids.clone();
                    for (int i = 0; i < _n_centroids; ++i) {
                        _centroids[i] = (closest == i).sum().item<bool>() ? x.index({closest == i}).mean(0) : _centroids[i];
                    }
                }
            }

            const torch::Tensor centroids() const { return _centroids; }

        private:
            int _n_centroids;
            KMeansInit _init;
            double _tol;
            torch::Tensor _centroids;

            void _plusplus(const torch::Tensor& x)

            {
                int i = 0;
                int cluster_idx = torch::randint(x.size(0), {1}).item<int>();
                _centroids[i++] = x[cluster_idx];
                for (; i < _n_centroids; ++i) {
                    torch::Tensor dist = std::get<0>(torch::min(torch::cdist(x, _centroids), -1));
                    cluster_idx = torch::multinomial(dist.pow(2), 1).item<int>();
                    _centroids[i] = x[cluster_idx];
                }
            }
        };
    } // namespace clustring
} // namespace butane
