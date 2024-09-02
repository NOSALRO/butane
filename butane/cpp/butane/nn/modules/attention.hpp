#pragma once

#include "../nn.hpp"

namespace butane {
    namespace nn {
        class AttentionImpl : public torch::nn::Module {
        public:
            AttentionImpl() = default;

            AttentionImpl(int d_model, int n_heads, double dropout = .0) : _d_model(d_model), _n_heads(n_heads)
            {
                _scale = std::pow(_d_model, -0.5);
                _key = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(false));
                _value = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(false));
                _query = torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(false));

                _dropout = torch::nn::Dropout(dropout);
                _lnorm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({_d_model}));

                register_module("Key", _key);
                register_module("Value", _value);
                register_module("Query", _query);
                register_module("Dropout", _dropout);
                register_module("LayerNorm", _lnorm);
            }

        private:
            int _d_model, _n_heads;
            double _scale = .0;
            torch::nn::Linear _key{nullptr}, _query{nullptr}, _value{nullptr};
            torch::nn::Dropout _dropout;
            torch::nn::LayerNorm _lnorm{nullptr};
        };

        TORCH_MODULE(Attention);
    } // namespace nn
} // namespace butane
