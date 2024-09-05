#pragma once

#include "../../types.hpp"
#include "../nn.hpp"
#include "../../clustering/kmeans.hpp"

namespace butane {
    namespace nn {
        class QuantizerImpl : public torch::nn::Module {
        public:
            torch::nn::Embedding embedding{nullptr};

            QuantizerImpl(
                int latent_dim,
                int n_centers,
                torch::Device device = torch::Device(torch::kCPU)) : _latent_dim(latent_dim), _n_centers(n_centers), _device(device)
            {
                embedding = torch::nn::Embedding(_n_centers, _latent_dim);
                embedding->weight.data().uniform_(-1. / static_cast<double>(_n_centers), 1. / static_cast<double>(_n_centers));
                embedding->weight.set_requires_grad(true);
                embedding = register_module("embeddings", embedding);
            }

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                torch::Tensor quantized_latents, closest_encoding;
                {
                    torch::NoGradGuard no_grad;
                    torch::Tensor distance = torch::sum(x.pow(2), 1, true) + \
                            torch::sum(embedding->weight.pow(2), 1) - \
                            2 * torch::matmul(x, embedding->weight.t());
                    closest_encoding = torch::argmin(distance, -1).squeeze();
                }
                quantized_latents = embedding->forward(closest_encoding);

                torch::Tensor commitment_loss = (quantized_latents.detach() - x).pow(2);
                torch::Tensor embedding_loss = (quantized_latents - x.detach()).pow(2);
                switch (_reduction) {
                case Mean:
                    commitment_loss = commitment_loss.mean(), embedding_loss = embedding_loss.mean();
                case Sum:
                    commitment_loss = commitment_loss.sum(), embedding_loss = embedding_loss.sum();
                }
                torch::Tensor quantization_loss = _beta * commitment_loss + embedding_loss;

                // Backprop trick. latent - latents = 0 however gradients are copied.
                quantized_latents = x + (quantized_latents - x).detach();
                return {quantized_latents, quantization_loss};
            }

            void init_codebook(double low, double high)
            {
                embedding->weight.data().uniform_(static_cast<double>(low), static_cast<double>(high));
                embedding->weight.set_requires_grad(true);
            }

            void init_codebook_kmeans(double low, double high, int max_data = -1)
            {
                max_data = (max_data == -1) ? _n_centers * 400 : max_data;
                torch::Tensor rdata = torch::empty({max_data, _latent_dim}).uniform_(static_cast<double>(low), static_cast<double>(high));
                rdata = rdata.to(_device);
                clustring::KMeans kmeans(_n_centers, butane::KMeansPlusPlus, 1e-4, -1);
                kmeans.fit(rdata);
                embedding->weight.set_data(kmeans.centroids());
                embedding->weight.set_requires_grad(true);
            }

            void set_beta(double new_beta)
            {
                _beta = new_beta;
            }

            void reduction(enum Reduction reduction_)
            {
                _reduction = reduction_;
            }

            void change_dev(torch::Device device)
            {
                _device = device;
                this->to(_device);
            }

            torch::Tensor centers() const { return embedding->weight; }
            double beta() const { return _beta; }

        protected:
            int _latent_dim, _n_centers;
            bool _constrained_latent_space;
            double _beta = 1.25;
            torch::Device _device = torch::Device(torch::kCPU);
            enum Reduction _reduction = Mean;
        };

        TORCH_MODULE(Quantizer);

        class Quantizer2dImpl : public QuantizerImpl {
        public:
            using QuantizerImpl::QuantizerImpl;

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
                x = x.permute({0, 2, 3, 1});
                torch::Tensor x_flat = x.clone().reshape({B * H * W, C});

                torch::Tensor quantized_latents, closest_encoding;
                {
                    torch::NoGradGuard no_grad;
                    torch::Tensor distance = torch::sum(x_flat.pow(2), 1, true) + \
                            torch::sum(embedding->weight.pow(2), 1) - \
                            2 * torch::matmul(x_flat, embedding->weight.t());
                    closest_encoding = torch::argmin(distance, -1).squeeze();
                }
                quantized_latents = embedding->forward(closest_encoding);
                quantized_latents = quantized_latents.view(x.sizes());

                torch::Tensor commitment_loss = (quantized_latents.detach() - x).pow(2);
                torch::Tensor embedding_loss = (quantized_latents - x.detach()).pow(2);
                switch (_reduction) {
                case Mean:
                    commitment_loss = commitment_loss.mean(), embedding_loss = embedding_loss.mean();
                case Sum:
                    commitment_loss = commitment_loss.sum(), embedding_loss = embedding_loss.sum();
                }
                torch::Tensor quantization_loss = _beta * commitment_loss + embedding_loss;

                // Backprop trick. latent - latents = 0 however gradients are copied.
                quantized_latents = x + (quantized_latents - x).detach();
                quantized_latents = quantized_latents.permute({0, 3, 1, 2});
                return {quantized_latents, quantization_loss};
            }
        };

        TORCH_MODULE(Quantizer2d);
    } // namespace nn
} // namespace butane
