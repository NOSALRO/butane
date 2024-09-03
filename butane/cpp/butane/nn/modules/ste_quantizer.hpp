#pragma once

#include "../../types.hpp"
#include "../nn.hpp"
#include <boost/optional.hpp>

namespace butane {
    namespace nn {
        class AffineTransformImpl : public torch::nn::Module {
        public:
            AffineTransformImpl() = default;
            AffineTransformImpl(int64_t latent_dims, bool use_running_statistics = false, double momentum = 0.1,
                double lr_scale = 1.0, int64_t num_groups = 1)
                : _use_running_statistics(use_running_statistics),
                  _num_groups(num_groups),
                  _momentum(momentum),
                  _lr_scale(lr_scale)
            {

                if (_use_running_statistics) {
                    _running_statistics_initialized = torch::zeros(1);
                    _running_ze_mean = register_buffer("running_ze_mean", torch::zeros({_num_groups, latent_dims}));
                    _running_ze_var = register_buffer("running_ze_var", torch::ones({_num_groups, latent_dims}));
                    _running_c_mean = register_buffer("running_c_mean", torch::zeros({_num_groups, latent_dims}));
                    _running_c_var = register_buffer("running_c_var", torch::ones({_num_groups, latent_dims}));
                }
                else {
                    _scale = register_parameter("scale", torch::zeros({_num_groups, latent_dims}));
                    _bias = register_parameter("bias", torch::zeros({_num_groups, latent_dims}));
                }
            }

            torch::Tensor forward(torch::Tensor embeddings)
            {
                auto [scale, bias] = get_affine_params();
                int64_t n = embeddings.size(0), c = embeddings.size(1);
                embeddings = embeddings.view({_num_groups, -1, c});
                embeddings = scale * embeddings + bias;
                return embeddings.view({n, c});
            }

            void update_running_statistics(torch::Tensor z_e, torch::Tensor c)
            {
                if (this->is_training() && _use_running_statistics) {
                    bool unbiased = false;

                    auto ze_mean = z_e.mean(c10::IntArrayRef{0, 1}).unsqueeze(0);
                    auto ze_var = z_e.var(c10::IntArrayRef{0, 1}, unbiased).unsqueeze(0);
                    auto c_mean = c.mean(0).unsqueeze(0);
                    auto c_var = c.var(0, unbiased).unsqueeze(0);

                    if (!_running_statistics_initialized.item<bool>()) {
                        _running_ze_mean.copy_(ze_mean);
                        _running_ze_var.copy_(ze_var);
                        _running_c_mean.copy_(c_mean);
                        _running_c_var.copy_(c_var);
                        _running_statistics_initialized.fill_(1);
                    }
                    else {
                        _running_ze_mean = (_momentum * ze_mean) + ((1 - _momentum) * _running_ze_mean);
                        _running_ze_var = (_momentum * ze_var) + ((1 - _momentum) * _running_ze_var);
                        _running_c_mean = (_momentum * c_mean) + ((1 - _momentum) * _running_c_mean);
                        _running_c_var = (_momentum * c_var) + ((1 - _momentum) * _running_c_var);
                    }
                }
            }

        private:
            torch::Tensor _running_statistics_initialized;
            torch::Tensor _running_ze_mean;
            torch::Tensor _running_ze_var;
            torch::Tensor _running_c_mean;
            torch::Tensor _running_c_var;

            torch::Tensor _scale;
            torch::Tensor _bias;

            bool _use_running_statistics;
            int64_t _num_groups;
            double _momentum;
            double _lr_scale;

            std::tuple<torch::Tensor, torch::Tensor> get_affine_params()
            {
                torch::Tensor scale, bias;
                if (_use_running_statistics) {
                    scale = (_running_ze_var / (_running_c_var + 1e-8)).sqrt();
                    bias = -scale * _running_c_mean + _running_ze_mean;
                }
                else {
                    scale = (1.0 + _lr_scale * this->_scale);
                    bias = _lr_scale * this->_bias;
                }
                return std::make_tuple(scale.unsqueeze(1), bias.unsqueeze(1));
            }
        };

        TORCH_MODULE(AffineTransform);

        template <typename Optimizer>
        class STEQuantizerImpl : public QuantizerImpl {
        public:
            STEQuantizerImpl(
                int latent_dim,
                int n_centers,
                torch::Device device = torch::Device(torch::kCPU),
                double sync_nu = 0.0,
                double affine_lr = 0.0,
                int affine_groups = 1,
                bool use_optimizer = false) : QuantizerImpl(latent_dim, n_centers, device), _sync_nu(sync_nu), _affine_lr(affine_lr), _affine_groups(affine_groups), _has_optimizer(use_optimizer)
            {
                if (_has_optimizer)
                    _optimizer = std::make_shared<Optimizer>(embedding->parameters());

                if (_affine_lr > 0.) {
                    _affine_transform = register_module("AffineTransform", AffineTransform(_latent_dim, false, _affine_lr, _affine_groups));
                    _has_affince_transform = true;
                }
            }

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                if (this->is_training() && _has_affince_transform) {
                    _affine_transform->update_running_statistics(x, embedding->weight);
                    embedding->weight.set_data(_affine_transform->forward(embedding->weight));
                }

                torch::Tensor quantized_latents, closest_encoding;
                {
                    torch::NoGradGuard no_grad;
                    torch::Tensor distance = torch::sum(x.pow(2), 1, true) + torch::sum(embedding->weight.pow(2), 1) - 2 * torch::matmul(x, embedding->weight.t());
                    closest_encoding = torch::argmin(distance, -1).squeeze();
                }
                quantized_latents = embedding->forward(closest_encoding);
                if (this->is_training() && _has_optimizer) {
                    assert((_beta == 1.0) && "Optimizer can only be used with beta=1.0");
                    switch (_reduction) {
                    case Sum:
                        ((quantized_latents - x.detach()).pow(2)).sum().backward();
                    case Mean:
                        ((quantized_latents - x.detach()).pow(2)).mean().backward();
                        _optimizer->step();
                        _optimizer->zero_grad();
                        quantized_latents = embedding->forward(closest_encoding);
                    }
                }

                torch::Tensor commitment_loss = (quantized_latents.detach() - x).pow(2);
                torch::Tensor embedding_loss = (quantized_latents - x.detach()).pow(2);
                switch (_reduction) {
                case Mean:
                    commitment_loss = commitment_loss.mean(), embedding_loss = embedding_loss.mean();
                case Sum:
                    commitment_loss = commitment_loss.sum(), embedding_loss = embedding_loss.sum();
                }
                torch::Tensor quantization_loss = _beta * commitment_loss + embedding_loss;

                if (_sync_nu > 0.)
                    quantized_latents = x + (quantized_latents - x).detach() + (_sync_nu * quantized_latents) + (-_sync_nu * quantized_latents).detach();
                else
                    quantized_latents = x + (quantized_latents - x).detach();

                return {quantized_latents, quantization_loss};
            }

            void set_sync_nu(double new_sync_nu)
            {
                _sync_nu = new_sync_nu;
            }

            void set_affine_lr(double new_affine_lr)
            {
                _affine_lr = new_affine_lr;
            }

            double sync_nu() const { return _sync_nu; }
            double affine_lr() const { return _affine_lr; }

        protected:
            double _sync_nu, _affine_lr;
            int _affine_groups;
            std::shared_ptr<Optimizer> _optimizer;
            AffineTransform _affine_transform;
            bool _has_optimizer = false, _has_affince_transform = false;
        };
        TORCH_MODULE_TEMPLATED(STEQuantizer);

        template <typename Optimizer>
        class STEQuantizer2dImpl : public STEQuantizerImpl<Optimizer> {
        public:
            using STEQuantizerImpl<Optimizer>::STEQuantizerImpl;

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
                x = x.permute({0, 2, 3, 1});
                torch::Tensor x_flat = x.clone().reshape({B * H * W, C});

                if (this->_has_affince_transform && this->is_training()) {
                    this->_affine_transform->update_running_statistics(x, this->embedding->weight);
                    this->embedding->weight.set_data(this->_affine_transform->forward(this->embedding->weight));
                }

                torch::Tensor quantized_latents, closest_encoding;
                {
                    torch::NoGradGuard no_grad;
                    torch::Tensor distance = torch::sum(x_flat.pow(2), 1, true) + torch::sum(this->embedding->weight.pow(2), 1) - 2 * torch::matmul(x_flat, this->embedding->weight.t());
                    closest_encoding = torch::argmin(distance, -1).squeeze();
                }
                quantized_latents = this->embedding->forward(closest_encoding);
                quantized_latents = quantized_latents.view(x.sizes());

                if (this->is_training() && this->_has_optimizer) {
                    assert((this->_beta == 1.0) && "Optimizer can only be used with beta=1.0");
                    switch (this->_reduction) {
                    case Sum:
                        ((quantized_latents - x.detach()).pow(2)).sum().backward();
                    case Mean:
                        ((quantized_latents - x.detach()).pow(2)).mean().backward();
                        this->_optimizer->step();
                        this->_optimizer->zero_grad();
                        quantized_latents = this->embedding->forward(closest_encoding);
                    }
                }
                quantized_latents = quantized_latents.view(x.sizes());

                torch::Tensor commitment_loss = (quantized_latents.detach() - x).pow(2);
                torch::Tensor embedding_loss = (quantized_latents - x.detach()).pow(2);
                switch (this->_reduction) {
                case Mean:
                    commitment_loss = commitment_loss.mean(), embedding_loss = embedding_loss.mean();
                case Sum:
                    commitment_loss = commitment_loss.sum(), embedding_loss = embedding_loss.sum();
                }
                torch::Tensor quantization_loss = this->_beta * commitment_loss + embedding_loss;

                if (this->_sync_nu > 0.)
                    quantized_latents = x + (quantized_latents - x).detach() + (this->_sync_nu * quantized_latents) + (-this->_sync_nu * quantized_latents).detach();
                else
                    quantized_latents = x + (quantized_latents - x).detach();

                quantized_latents = quantized_latents.permute({0, 3, 1, 2});
                return {quantized_latents, quantization_loss};
            }
        };
        TORCH_MODULE_TEMPLATED(STEQuantizer2d);
    } // namespace nn
} // namespace butane
