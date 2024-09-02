#include "butane/butane.h"

int main(int argc, char** argv)
{
    // butane::nn::Attention att(512, 8, .3);
    torch::nn::TransformerEncoderLayer l(torch::nn::TransformerEncoderLayerOptions(512, 8));
    torch::nn::TransformerEncoder att(torch::nn::TransformerEncoderOptions(l, 1));
    std::cout << att << std::endl;
    return 0;
}
