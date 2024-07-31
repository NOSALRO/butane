#pragma once

#include "dataset.hpp"

namespace butane {
    namespace data {
        namespace ops {
            void drop(Dataset& dataset, double prec)
            {
                prec = 1. - prec;
                torch::Tensor& _data = dataset.data_ref();
                torch::Tensor& _targets = dataset.targets_ref();
                if (prec == 0) {
                    _data = torch::Tensor();
                    return;
                }
                int numel_to_keep = std::floor(dataset.size() * prec);
                torch::Tensor idx_to_keep = torch::randperm(dataset.size()).index({torch::arange(numel_to_keep)});
                _data = _data.index({idx_to_keep});
                _targets = !_targets.numel() ? _targets.index({idx_to_keep}) : torch::Tensor();
            }

            void drop_to_max_size(Dataset& dataset, int max_size)
            {
                if (max_size > dataset.size())
                    return;
                torch::Tensor& _data = dataset.data_ref();
                torch::Tensor& _targets = dataset.targets_ref();
                torch::Tensor idx_to_keep = torch::randperm(dataset.size()).index({torch::arange(max_size)});
                _data = _data.index({idx_to_keep});
                _targets = !_targets.numel() ? _targets.index({idx_to_keep}) : torch::Tensor();
            }

            void sparsify(Dataset& dataset)
            {
                double prev_mu = 0;
                torch::Tensor& _data = dataset.data_ref();
                torch::Tensor& _targets = dataset.targets_ref();
                while (1) {
                    torch::Tensor distances = torch::cdist(_data.flatten(1), _data.flatten(1));
                    double mu = distances.mean().item<double>();
                    if (mu - prev_mu < 1e-02) {
                        break;
                    }
                    prev_mu = mu;
                    int denser = torch::argmin(torch::sum(distances, 1)).item<int>();
                    _data = torch::vstack({_data.slice(0, 0, denser), _data.slice(0, denser + 1, dataset.size())});
                    if (!_targets.numel()) {
                        _targets = torch::cat({_targets.slice(0, 0, denser), _targets.slice(0, denser + 1, dataset.size())}, 0);
                    }
                }
            }
        } // namespace ops
    } // namespace data
} // namespace butane
