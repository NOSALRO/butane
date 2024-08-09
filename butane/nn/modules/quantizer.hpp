#pragma once

#include "../nn.hpp"
#include <kmeans/kmeans.hpp>

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
                embedding->weight.data().uniform_(-1. / static_cast<double>(_n_centers), 1. /static_cast<double>(_n_centers));
                embedding->weight.set_requires_grad(true);
                embedding = register_module("embeddings", embedding);
            }

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                torch::Tensor quantized_latents;
                torch::Tensor distance = torch::cdist(x, embedding->weight);
                torch::Tensor closest_encoding = torch::argmin(distance, -1).unsqueeze(1);

                torch::Tensor encoding_one_hot = torch::zeros({closest_encoding.sizes()[0], _n_centers}, torch::TensorOptions().dtype(torch::kFloat32)).to(x.device());
                encoding_one_hot = encoding_one_hot.scatter_(1, closest_encoding, 1);

                quantized_latents = torch::matmul(encoding_one_hot, embedding->weight);
                quantized_latents = quantized_latents.view(x.sizes());

                torch::Tensor commitment_loss = (quantized_latents.detach() - x).pow(2).mean();
                torch::Tensor embedding_loss = (quantized_latents - x.detach()).pow(2).mean();

                // Backprop trick. latent - latents = 0 however gradients are copied.
                torch::Tensor quantization_loss = _beta * commitment_loss + embedding_loss;
                quantized_latents = x + (quantized_latents - x).detach();
                return {quantized_latents, quantization_loss};
            }

            void init_codebook(double low, double high)
            {
                embedding->weight.data().uniform_(static_cast<double>(low), static_cast<double>(high));
                embedding->weight.set_requires_grad(true);
            }

            void init_codebook_kmeans(double low, double high)
            {
                torch::Tensor rdata = torch::empty({_n_centers * 400, _latent_dim}).uniform_(static_cast<double>(low), static_cast<double>(high));
                rdata = rdata.to(_device);
                torch_kmeans::Kmeans kmeans(_n_centers, "kmeans++", 1e-18, -1);
                kmeans.fit(rdata);
                embedding->weight.set_data(kmeans.clusters());
                embedding->weight.set_requires_grad(true);
            }

            void set_beta(double new_beta)
            {
                _beta = new_beta;
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
        };

        TORCH_MODULE(Quantizer);

        class Quantizer2dImpl : public QuantizerImpl {
        public:
            using QuantizerImpl::QuantizerImpl;

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
                x = x.permute({0, 2, 3, 1}).reshape({B * H * W, C});
                torch::Tensor quantized_latents;
                torch::Tensor distance = torch::cdist(x, embedding->weight);
                torch::Tensor closest_encoding = torch::argmin(distance, -1).unsqueeze(1);

                torch::Tensor encoding_one_hot = torch::zeros({closest_encoding.sizes()[0], _n_centers}, torch::TensorOptions().dtype(torch::kFloat32)).to(x.device());
                encoding_one_hot = encoding_one_hot.scatter_(1, closest_encoding, 1);

                quantized_latents = torch::matmul(encoding_one_hot, embedding->weight);
                quantized_latents = quantized_latents.view(x.sizes());

                torch::Tensor commitment_loss = (quantized_latents.detach() - x).pow(2).mean();
                torch::Tensor embedding_loss = (quantized_latents - x.detach()).pow(2).mean();

                // Backprop trick. latent - latents = 0 however gradients are copied.
                torch::Tensor quantization_loss = _beta * commitment_loss + embedding_loss;
                quantized_latents = x + (quantized_latents - x).detach();
                quantized_latents = quantized_latents.reshape({B, H, W, C});
                quantized_latents = quantized_latents.permute({0, 3, 1, 2});
                return {quantized_latents, quantization_loss};
            }
        };

        TORCH_MODULE(Quantizer2d);
    } // namespace nn
} // namespace butane
