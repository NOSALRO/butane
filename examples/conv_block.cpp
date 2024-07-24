#include "butane/butane.h"

using namespace butane::nn;

int main(int argc, char** argv)
{
    Conv1dBlockImpl<torch::nn::InstanceNorm1d, torch::nn::MaxPool1d>::Config co{
        .input_dims = {2, 100},
        .channels = {32, 128},
        .activation_function = {torch::nn::ReLU(), torch::nn::GELU()},
        .conv_kernels = {{2}, {1}},
        .conv_stride = {{1}, {2}},
        .normalization = {true}
    };
    Conv1dBlock<torch::nn::InstanceNorm1d, torch::nn::MaxPool1d> c1(co);
    // Conv1dBlock<torch::nn::InstanceNorm1d, torch::nn::MaxPool1d> c1(
    //     std::vector<int64_t>{2, 100},
    //     std::vector<int64_t>{64, 64},
    //     AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
    //     std::vector<std::vector<int64_t>>{{1}, {2}}
    // );

    ConvTranspose1dBlock<torch::nn::InstanceNorm1d, torch::nn::MaxPool1d> c1t(
        std::vector<int64_t>{2, 100},
        std::vector<int64_t>{64, 64},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1}, {2}}
    );
    std::cout << c1 << std::endl;
    std::cout << c1t << std::endl;

    Conv2dBlock<torch::nn::InstanceNorm2d, torch::nn::MaxPool2d> c2(
        std::vector<int64_t>{2, 100},
        std::vector<int64_t>{64, 64},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1}, {2, 3}}
    );
    ConvTranspose2dBlock<torch::nn::InstanceNorm2d, torch::nn::MaxPool2d> c2t(
        std::vector<int64_t>{2, 100},
        std::vector<int64_t>{64, 64},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1}, {2, 3}}
    );
    std::cout << c2 << std::endl;
    std::cout << c2t << std::endl;

    Conv3dBlock<torch::nn::InstanceNorm3d, torch::nn::MaxPool3d> c3(
        std::vector<int64_t>{2, 100},
        std::vector<int64_t>{64, 64},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1, 2, 3}, {2}}
    );

    ConvTranspose3dBlock<torch::nn::InstanceNorm3d, torch::nn::MaxPool3d> c3t(
        std::vector<int64_t>{2, 100},
        std::vector<int64_t>{64, 64},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1, 2, 3}, {2}}
    );

    std::cout << c3 << std::endl;
    std::cout << c3t << std::endl;
    return 0;
}
