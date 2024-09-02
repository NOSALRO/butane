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
                _data = data.detach().clone();
                _data = _data.toType(torch::kFloat32);
                if (targets) {
                    _targets = targets.value().detach().clone();
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

            void set(const torch::Tensor& data, boost::optional<const torch::Tensor&> targets = boost::none)
            {
                _data = data.detach().clone().to(_device);
                if (targets) {
                    _targets = targets.value().detach().clone().to(_device);
                    _has_targets = true;
                }
            }

            void append(const torch::Tensor& data, boost::optional<const torch::Tensor&> targets = boost::none)
            {
                if (_data.sizes()[0] == 0) {
                    _data = data.detach().clone();
                }
                else {
                    torch::Tensor tmp_data = data.detach().clone();
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
                        _targets = targets.value().detach().clone();
                    else {
                        torch::Tensor tmp_targets = targets.value().detach().clone();
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
            const std::vector<int64_t> sizes() const { return _data.sizes().vec(); }
            const int size() const { return _data.size(0); }
            int numel() const { return torch::tensor(_data.sizes()).prod().item<int>(); }

            torch::Tensor& data_ref() { return _data; }
            torch::Tensor& targets_ref() { return _targets; }
            torch::Tensor data() { return _data.detach().clone(); }
            torch::Tensor targets() { return _targets.detach().clone(); }

        protected:
            torch::Tensor _data, _targets;
            bool _has_targets = false;
            int _numel = 0;
            torch::Device _device = torch::kCPU;
        };
    } // namespace data
} // namespace butane
