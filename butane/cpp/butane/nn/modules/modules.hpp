#pragma once

#include "../nn.hpp"

namespace butane {
    namespace nn {
        namespace functional {
            inline torch::Tensor gaussian(torch::Tensor x)
            {
                return (-x.square()).exp();
            }

            inline torch::Tensor squashing(torch::Tensor x)
            {
                return (9 / 8. * torch::sin(x)) + (1 / 8. * torch::sin(3. * x));
            }

            inline torch::nn::AnyModule unflatten(int64_t start_dim, const torch::Tensor& sizes)
            {
                std::vector<int64_t> vec_sizes;
                vec_sizes.reserve(sizes.numel());
                for (unsigned int i = 0; i < sizes.numel(); ++i)
                    vec_sizes.push_back(sizes[i].item<int64_t>());
                return torch::nn::AnyModule(torch::nn::Unflatten(torch::nn::UnflattenOptions(1, vec_sizes)));
            }

            inline torch::nn::AnyModule flatten(int64_t start_dim)
            {
                return torch::nn::AnyModule(torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)));
            }
        } // namespace functional

        class GaussianImpl : public torch::nn::Module {
        public:
            GaussianImpl() = default;

            inline torch::Tensor forward(torch::Tensor x)
            {
                return functional::gaussian(x);
            }
        };
        TORCH_MODULE(Gaussian);

        class SquashingImpl : public torch::nn::Module {
        public:
            SquashingImpl() = default;

            inline torch::Tensor forward(torch::Tensor x)
            {
                return functional::squashing(x);
            }
        };
        TORCH_MODULE(Squashing);
    } // namespace nn
} // namespace butane
