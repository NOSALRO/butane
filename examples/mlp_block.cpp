#include "butane/butane.h"

int main(int argc, char** argv)
{
    butane::nn::MLPBlock mlp(butane::nn::MLPBlockImpl::Config{
        .input_dims = 2,
        .output_dims = 3,
        .hidden_dims = std::vector<int64_t>{64, 64},
        .activation_function =butane::nn::AnyModuleList({torch::nn::ReLU(), torch::nn::ReLU()}),
        .output_activation = false,
        .bias = std::vector<bool>{false}}
    );
    std::cout << mlp << std::endl;
    return 0;
}
