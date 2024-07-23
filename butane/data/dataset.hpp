#pragma once

#include <boost/optional.hpp>
#include <random>
#include <torch/torch.h>

namespace data {

    class Dataset {

    public:
        Dataset() = default;

        Dataset(torch::Tensor data, boost::optional<torch::Tensor> targets = boost::none)
        {
            _data = data.clone();
            _data = _data.toType(torch::kFloat32);
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
            if (targets.has_value()) {
                _targets = targets.value().clone();
                _has_targets = true;
            }
        }

        Dataset(const std::string& fp)
        {
            torch::load(_data, fp);
            _data = _data.toType(torch::kFloat32);
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        struct Batch {
            torch::Tensor data;
            torch::Tensor target;
            Batch(torch::Tensor data_, torch::Tensor target_) : data(data_), target(target_) {}
        };

        Batch get(size_t index)
        {
            return {_data[index].clone(), (_targets.numel() == 0) ? torch::Tensor() : _targets[index].clone()};
        }

        Batch get(torch::Tensor index)
        {
            return Batch(_data.index({index}).clone(), (_targets.numel() == 0) ? torch::Tensor() : _targets.index({index}).clone());
        }

        Batch get(std::vector<int> index)
        {
            torch::Tensor tensor_idx = torch::from_blob(index.data(), {static_cast<int>(index.size())}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kInt64);
            return Batch(_data.index({tensor_idx}).clone(), (_targets.numel() == 0) ? torch::Tensor() : _targets.index({tensor_idx}).clone());
        }

        void shuffle()
        {
            torch::Tensor random_ordered_indexes = torch::randperm(_data.size(0));
            _data = _data.index({random_ordered_indexes});
            _targets = _has_targets ? _targets.index({random_ordered_indexes}) : _targets;
        }

        void flatten()
        {
            _data = torch::flatten(_data, 1);
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        torch::Tensor operator[](int idx) const { return _data[idx]; }

        void to(torch::Device device)
        {
            _data = _data.to(device);
            if (_has_targets)
                _targets = _targets.to(device);
            _device = device;
        }

        void save(const std::string& fp) const
        {
            std::string fpath = fp.substr(0, fp.rfind(".")); // token is "scott"
            torch::save(_data.clone().cpu().detach(), fpath + "_data.pt");
            if (_has_targets)
                torch::save(_targets.clone().cpu().detach(), fpath + "_targets.pt");
        }
        void set_data(const torch::Tensor& data)
        {
            _data = data.clone();
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        void set_targets(const torch::Tensor& targets) { _targets = targets.clone(); }

        void append(const torch::Tensor& data, boost::optional<const torch::Tensor&> targets = boost::none)
        {
            if (_data.sizes()[0] == 0) {
                _data = data.clone();
            }
            else {
                torch::Tensor tmp_data = data.clone();
                if (_data.sizes().size() > tmp_data.sizes().size()) {
                    tmp_data = tmp_data.unsqueeze(0);
                }
                else if (_data.sizes().size() < tmp_data.sizes().size()) {
                    throw std::invalid_argument("Wrong data sizes!");
                }
                _data = torch::vstack({_data, tmp_data.clone()});
            }
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());

            if (targets.has_value()) {
                _has_targets = true;
                if (_targets.numel() == 0)
                    _targets = targets.value().clone();
                else {
                    torch::Tensor tmp_targets = targets.value().clone();
                    _targets = torch::cat({_targets, tmp_targets}, 0);
                }
            }
        }

        void drop(double prec)
        {
            prec = 1. - prec;
            if (prec == 0) {
                _data = torch::Tensor();
                _data_size.clear();
                return;
            }
            int numel_to_keep = std::floor(_data_size[0] * prec);
            torch::Tensor idx_to_keep = torch::randperm(_data_size[0]).index({torch::arange(numel_to_keep)});
            _data = _data.index({idx_to_keep});
            _targets = _has_targets ? _targets.index({idx_to_keep}) : _targets;
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        void drop_to_max_size(int max_size)
        {
            if (max_size > _data_size[0])
                return;
            torch::Tensor idx_to_keep = torch::randperm(_data_size[0]).index({torch::arange(max_size)});
            _data = _data.index({idx_to_keep});
            _targets = _has_targets ? _targets.index({idx_to_keep}) : _targets;
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        void sparsify()
        {
            // double prev_mu = std::numeric_limits<double>::max();
            double prev_mu = 0;
            // for (unsigned int i = 0; i < std::abs(_data_size[0] - 2000); i++) {
            while (1) {
                torch::Tensor distances = torch::cdist(_data.flatten(1), _data.flatten(1));
                double mu = distances.mean().item<double>();
                if (mu - prev_mu < 1e-02) {
                    break;
                }
                prev_mu = mu;
                int denser = torch::argmin(torch::sum(distances, 1)).item<int>();
                _data = torch::vstack({_data.slice(0, 0, denser), _data.slice(0, denser + 1, _data.sizes()[0])});
                if (_has_targets) {
                    _targets = torch::cat({_targets.slice(0, 0, denser), _targets.slice(0, denser + 1, _targets.sizes()[0])}, 0);
                }
            }
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        const std::vector<int64_t>& sizes() const { return _data_size; }
        torch::optional<size_t> size() const { return _data.size(0); }
        torch::Tensor data() { return _data.clone(); }
        torch::Tensor targets() { return _targets.clone(); }
        int numel() { return _numel; }

    protected:
        torch::Tensor _data, _targets;
        std::vector<int64_t> _data_size;
        bool _has_targets = false;
        int _numel = 0;
        torch::Device _device = torch::kCPU;
    };
} // namespace data
