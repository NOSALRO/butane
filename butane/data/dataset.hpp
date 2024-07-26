#pragma once

#include <any>
#include <boost/optional.hpp>
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

namespace butane {
    namespace data {

        struct Batch {
            torch::Tensor data;
            torch::Tensor target;
            Batch(torch::Tensor data_, torch::Tensor target_) : data(data_), target(target_) {}
        };

        class Dataset {
        public:
            Dataset() = default;

            Dataset(const torch::Tensor& data, boost::optional<const torch::Tensor&> targets = boost::none)
            {
                _data = data.clone();
                _data = _data.toType(torch::kFloat32);
                if (targets) {
                    _targets = targets.value().clone();
                    _has_targets = true;
                }
            }

            Dataset(const std::string& fp_data, const std::string& fp_target = "")
            {
                torch::load(_data, fp_data);
                _data = _data.toType(torch::kFloat32);

                if (!(fp_target == "")) {
                    torch::load(_targets, fp_target);
                    _has_targets = true;
                }
            }

            Batch get(size_t index)
            {
                return {_data[index], (_targets.numel() == 0) ? torch::Tensor() : _targets[index]};
            }

            Batch get(torch::Tensor index)
            {
                return Batch(_data.index({index}), (_targets.numel() == 0) ? torch::Tensor() : _targets.index({index}));
            }

            Batch get(std::vector<int>& index)
            {
                torch::Tensor tensor_idx = torch::from_blob(index.data(), {static_cast<int>(index.size())}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kInt64);
                return Batch(_data.index({tensor_idx}), (_targets.numel() == 0) ? torch::Tensor() : _targets.index({tensor_idx}));
            }

            void flatten(int dim = 1) { _data = torch::flatten(_data, dim); }

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
                    _data = torch::vstack({_data, tmp_data});
                }
                if (targets) {
                    _has_targets = true;
                    if (_targets.numel() == 0)
                        _targets = targets.value().clone();
                    else {
                        torch::Tensor tmp_targets = targets.value().clone();
                        _targets = torch::cat({_targets, tmp_targets}, 0);
                    }
                }
            }

            void move(torch::Tensor& data, boost::optional<torch::Tensor&> targets = boost::none)
            {
                if (size() == 0) {
                    _data = std::move(data);
                }
                else {
                    torch::Tensor tmp_data = std::move(data);
                    if (sizes().size() > tmp_data.sizes().size()) {
                        tmp_data = tmp_data.unsqueeze(0);
                    }
                    else if (sizes().size() < tmp_data.sizes().size()) {
                        throw std::invalid_argument("Wrong data sizes!");
                    }
                    _data = torch::vstack({_data, std::move(tmp_data)});
                }

                if (targets) {
                    _has_targets = true;
                    if (_targets.numel() == 0)
                        _targets = std::move(targets.value());
                    else {
                        torch::Tensor tmp_targets = std::move(targets.value());
                        _targets = torch::cat({_targets, tmp_targets}, 0);
                    }
                }
            }

            void drop(double prec)
            {
                prec = 1. - prec;
                if (prec == 0) {
                    _data = torch::Tensor();
                    _data_size.empty();
                    return;
                }
                int numel_to_keep = std::floor(size() * prec);
                torch::Tensor idx_to_keep = torch::randperm(size()).index({torch::arange(numel_to_keep)});
                _data = _data.index({idx_to_keep});
                _targets = _has_targets ? _targets.index({idx_to_keep}) : _targets;
            }

            void drop_to_max_size(int max_size)
            {
                if (max_size > size())
                    return;
                torch::Tensor idx_to_keep = torch::randperm(size()).index({torch::arange(max_size)});
                _data = _data.index({idx_to_keep});
                _targets = _has_targets ? _targets.index({idx_to_keep}) : _targets;
            }

            void sparsify()
            {
                double prev_mu = 0;
                while (1) {
                    torch::Tensor distances = torch::cdist(_data.flatten(1), _data.flatten(1));
                    double mu = distances.mean().item<double>();
                    if (mu - prev_mu < 1e-02) {
                        break;
                    }
                    prev_mu = mu;
                    int denser = torch::argmin(torch::sum(distances, 1)).item<int>();
                    _data = torch::vstack({_data.slice(0, 0, denser), _data.slice(0, denser + 1, size())});
                    if (_has_targets) {
                        _targets = torch::cat({_targets.slice(0, 0, denser), _targets.slice(0, denser + 1, size())}, 0);
                    }
                }
            }

            const std::vector<int64_t> sizes() const { return _data.sizes().vec(); }
            const int size() const { return _data.size(0); }
            int numel() const { return torch::tensor(_data.sizes()).prod().item<int>(); }

            torch::Tensor& data_ref() { return _data; }
            torch::Tensor& targets_ref() { return _targets; }
            torch::Tensor data() { return _data.clone(); }
            torch::Tensor targets() { return _targets.clone(); }

        protected:
            torch::Tensor _data, _targets;
            c10::ArrayRef<int64_t> _data_size;
            bool _has_targets = false;
            int _numel = 0;
            torch::Device _device = torch::kCPU;
        };
    } // namespace data
} // namespace butane
