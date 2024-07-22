#pragma once

#include <boost/optional.hpp>
#include <random>
#include <torch/torch.h>

namespace data {
    class Dataset : public torch::data::datasets::Dataset<Dataset> {
        using Example = torch::data::Example<>;

    public:
        Dataset() = default;

        Dataset(const torch::Tensor& data, boost::optional<torch::Tensor> targets = boost::none)
        {
            _data = data.clone();
            _data = _data.toType(torch::kFloat32);
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
            if (targets.has_value())
                _targets = targets.value().clone();
        }

        Dataset(const std::string& fp)
        {
            torch::load(_data, fp);
            _data = _data.toType(torch::kFloat32);
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        torch::data::Example<> get(size_t index) override
        {
            return {_data[index].clone(), _targets[index].clone()}; //(_targets.numel() == 0) ? torch::Tensor() : _targets[index]);
        }

        void shuffle()
        {
            auto rng = std::default_random_engine{};
            std::vector<int> _idxes(_data_size[0], 0);
            std::iota(_idxes.begin() + 1, _idxes.end(), 1);
            std::shuffle(_idxes.begin(), _idxes.end(), rng);
            torch::Tensor shuffled = torch::empty(_data_size);
            for (int i = 0; i < _data_size[0]; i++)
                shuffled[i] = _data[_idxes[i]];
            _data = shuffled.view(_data.sizes()).to(_device);
        }

        void flatten()
        {
            _data = torch::flatten(_data, 1);
            _data_size = std::vector<int64_t>(_data.sizes().begin(), _data.sizes().end());
            _numel = std::accumulate(std::begin(_data_size), std::end(_data_size), 1.0, std::multiplies<int>());
        }

        torch::Tensor operator[](int idx)
        {
            return _data[idx];
        }

        void to(torch::Device device);
        void save(const std::string& fp) const;
        void set_data(const torch::Tensor& data);

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
                if (_targets.size(0) == 0) {
                    _targets = targets.value().clone();
                }
                else {
                    torch::Tensor tmp_targets = targets.value().clone();
                    if (_targets.sizes().size() == 1)
                        _targets = torch::hstack({_targets, tmp_targets.clone()});
                    else
                        _targets = torch::vstack({_targets, tmp_targets.clone()});
                }
            }
        }

        void drop(double prec);
        void drop_to_max_size(int max_size);
        void sparsify();

        // const std::vector<int64_t>& sizes() const { return _data_size; }
        torch::optional<size_t> size() const override { return _data.size(0); }
        torch::Tensor data() { return _data.clone(); }
        torch::Tensor targets() { return _targets.clone(); }
        int numel() { return _numel; }

    protected:
        torch::Tensor _data, _targets;
        std::vector<int64_t> _data_size;
        int _numel = 0;
        torch::Device _device = torch::kCPU;
    };
} // namespace data
