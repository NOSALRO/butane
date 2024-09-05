#pragma once

#include "../types.hpp"
#include <limits>
#include <torch/torch.h>

using namespace torch::indexing;
namespace butane {
    namespace clustring {
        class KMeans {
        public:
            KMeans(
                int n_centroids,
                KMeansInit init = KMeansPlusPlus,
                int max_iters = 500,
                double tol = 1e-04,
                int random_state = -1) : _n_centroids(n_centroids), _init(init), _max_iters(max_iters), _tol(tol)
            {
                random_state == -1 ? torch::cuda::manual_seed(time(0)) : torch::cuda::manual_seed(random_state);
                random_state == -1 ? torch::manual_seed(time(0)) : torch::manual_seed(random_state);
            };

            void fit(const torch::Tensor& x)
            {
                _centroids = torch::empty({_n_centroids, x.size(1)}).to(x.device());
                switch (_init) {
                case KMeansPlusPlus:
                    this->_init_plusplus(x);
                case Random:
                    this->_init_random(x);
                }

                torch::Tensor _prev_centroids;
                for (int64_t i = 0; i < _max_iters; ++i) {
                    _prev_centroids = _centroids.clone();
                    torch::Tensor dist = torch::cdist(x, _centroids);
                    torch::Tensor closest = dist.argmin(-1);
                    torch::Tensor assignments_per_centroid_count = torch::bincount(closest, {}, _n_centroids);
                    torch::Tensor sum_of_x_per_centroid = torch::zeros_like(_centroids).scatter_add_(0, closest.unsqueeze(1).expand_as(x), x);
                    torch::Tensor valid_centroids = assignments_per_centroid_count.nonzero().squeeze().to(x.device());
                    _centroids.index_put_(
                        {valid_centroids},
                        sum_of_x_per_centroid.index_select(0, {valid_centroids}) / assignments_per_centroid_count.index_select(0, {valid_centroids}).unsqueeze(1));
                    if (torch::linalg_norm(_centroids - _prev_centroids).item<double>() < _tol)
                        break;
                }
            }

            const torch::Tensor centroids() const { return _centroids; }

        protected:
            int _n_centroids;
            KMeansInit _init;
            int _max_iters;
            double _tol;
            torch::Tensor _centroids;

            void _init_random(const torch::Tensor& x)
            {
                torch::Tensor high_, low_;
                high_ = std::get<0>(torch::max(x, 0)), low_ = std::get<0>(torch::min(x, 0));
                for (int i = 0; i < x.sizes()[1]; ++i) {
                    _centroids.index({Slice(), i}).uniform_(low_[i].item<double>(), high_[i].item<double>());
                }
            }

            void _init_plusplus(const torch::Tensor& x)
            {
                int centroid_idx = torch::randint(x.size(0), {1}).item<int>();
                torch::Tensor track_selected = torch::zeros({x.size(0)}).to(torch::kBool);
                track_selected[centroid_idx] = true;
                torch::Tensor distances = torch::zeros({x.size(0), _n_centroids}).to(x.device());

                for (int64_t i = 0; i < _n_centroids - 1; ++i) {
                    distances.index_put_({Slice(), i}, torch::cdist(x, x[centroid_idx].unsqueeze(0)).pow(2).squeeze());
                    centroid_idx = torch::multinomial(std::get<0>(torch::min(distances.index({Slice(), Slice(None, i + 1)}), -1)), 1).item<int>();
                    track_selected[centroid_idx] = true;
                }
                _centroids = x.index_select(0, track_selected.nonzero().squeeze().to(x.device()));
            }
        };

        class MiniBatchKMeans : public KMeans{
        public:
            MiniBatchKMeans(
                int n_centroids,
                KMeansInit init = KMeansPlusPlus,
                int batch_size = 1024,
                int max_iters = 500,
                double tol = 1e-04,
                int random_state = -1) : KMeans(n_centroids, init, max_iters, tol), _batch_size(batch_size) {}

            void fit(const torch::Tensor& x)
            {
                _centroids = torch::empty({_n_centroids, x.size(1)}).to(x.device());
                switch (_init) {
                case KMeansPlusPlus:
                    this->_init_plusplus(x);
                case Random:
                    this->_init_random(x);
                }

                torch::Tensor _prev_centroids;
                for (int64_t i = 0; i < _max_iters; ++i) {
                    _prev_centroids = _centroids.clone();
                    torch::Tensor mini_batch_idx = torch::randint(0, x.size(0), {_batch_size}).to(x.device());
                    torch::Tensor mini_batch = x.index_select(0, {mini_batch_idx});

                    torch::Tensor dist = torch::cdist(mini_batch, _centroids);
                    torch::Tensor closest = dist.argmin(-1);
                    torch::Tensor assignments_per_centroid_count = torch::bincount(closest, {}, _n_centroids);
                    torch::Tensor sum_of_x_per_centroid = torch::zeros_like(_centroids).scatter_add_(0, closest.unsqueeze(1).expand_as(mini_batch), mini_batch);
                    torch::Tensor valid_centroids = assignments_per_centroid_count.nonzero().squeeze().to(x.device());
                    assignments_per_centroid_count = assignments_per_centroid_count.to(torch::kFloat32);
                    torch::Tensor eta = (1./assignments_per_centroid_count.index({valid_centroids})).unsqueeze(1);
                    _centroids.index_put_(
                        {valid_centroids},
                        ((1 - eta) * _centroids.index({valid_centroids}) +
                        (eta * sum_of_x_per_centroid.index({valid_centroids}) /
                        assignments_per_centroid_count.index({valid_centroids}).unsqueeze(1))));
                    if (torch::linalg_norm(_centroids - _prev_centroids).item<double>() < _tol)
                        break;
                }
            }

        private:
            int _batch_size;
        };
    } // namespace clustring
} // namespace butane
