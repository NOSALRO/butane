#pragma once

#include "nn.hpp"

namespace butane {
    namespace nn {
        class MLPBlockImpl : public torch::nn::Module {
        public:
            using ivec = dtypes::ivec;
            using dvec = dtypes::dvec;

            MLPBlockImpl() = default;

            MLPBlockImpl(
                int input_dims,
                int output_dims,
                const ivec& hidden_dims,
                AnyModuleList activation_function = AnyModuleList({torch::nn::ReLU()}),
                bool output_activation = false,
                dvec dropout = dvec{0.0}) : _input_dims(input_dims), _output_dims(output_dims), _hidden_dims(hidden_dims)
            {
                _hidden_dims.insert(_hidden_dims.begin(), _input_dims);
                _hidden_dims.insert(_hidden_dims.end(), _output_dims);
                size_t n_layers = _hidden_dims.size() - 1;

                assert((activation_function.size() == n_layers || activation_function.size() == 1 || (activation_function.size() == n_layers - 1 && !output_activation)) && "Activation Functions size should be 1 or N layers");
                assert((dropout.size() == n_layers || dropout.size() == 1) && "Dropout size should be 1 or N layers");

                utils::_fill_defaults(activation_function, n_layers);
                utils::_fill_defaults(dropout, n_layers);

                if (!output_activation) {
                    if (activation_function.size() == n_layers - 1)
                        activation_function.push_back(torch::nn::Identity());
                    else
                        activation_function.set(activation_function.size() - 1, torch::nn::Identity());
                }

                for (unsigned int i = 0; i < _hidden_dims.size() - 1; ++i) {
                    _mlp->push_back(torch::nn::Linear(_hidden_dims[i], _hidden_dims[i + 1]));
                    _mlp->push_back(activation_function[i]);
                    if (!(dropout[i] == 0.))
                        _mlp->push_back(torch::nn::Dropout(dropout[i]));
                }

                register_module("MLP", _mlp);
            }

            torch::Tensor forward(torch::Tensor x)
            {
                return _mlp->forward(x);
            }

            const int out_features() const { return _output_dims; }

        private:
            int _input_dims, _output_dims;
            ivec _hidden_dims;
            torch::nn::Sequential _mlp;
        };
        TORCH_MODULE(MLPBlock);
    } // namespace nn
} // namespace butane
