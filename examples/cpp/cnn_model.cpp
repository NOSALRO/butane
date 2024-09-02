#include "butane/butane.h"

using namespace butane::nn;

int main(int argc, char** argv)
{
    Conv2dBlock<torch::nn::InstanceNorm2d, torch::nn::MaxPool2d> c2(
        std::vector<int64_t>{2, 10, 10},
        std::vector<int64_t>{16, 16},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1}, {2, 3}}
    );


    MLPBlock mlp( c2->output_size().prod().item<int>(), 3, std::vector<int64_t>{64, 64});

    torch::nn::Sequential model(
        c2,
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(1)),
        mlp
    );

    std::cout << model << std::endl;

    std::cout << model->forward(torch::rand({1, 2, 10, 10})) << std::endl;

    return 0;
}
