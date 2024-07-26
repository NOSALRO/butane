#include "butane/butane.h"

int main(int argc, char** argv)
{
    butane::nn::MLPBlock mlp(2, 3, std::vector<int64_t>{64, 64}, butane::nn::AnyModuleList({torch::nn::ReLU(), torch::nn::ReLU()}));
    std::cout << mlp << std::endl;
    return 0;
}
