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

            inline torch::Tensor scaled_tanh(torch::Tensor x, double alpha = 1.)
            {
                return static_cast<double>(alpha) * torch::tanh(x);
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

        class ScaledTanhImpl : public torch::nn::Module {
        public:
            ScaledTanhImpl(double alpha = 1.) : _alpha(static_cast<double>(alpha)) {}

            inline torch::Tensor forward(torch::Tensor x)
            {
                std::cout << functional::scaled_tanh(x, _alpha).max() << std::endl;
                return functional::scaled_tanh(x, _alpha);
            }
        private:
            double _alpha;
        };
        TORCH_MODULE(ScaledTanh);
    } // namespace nn
} // namespace butane
