#include "butane/butane.h"

using namespace butane::nn;


int main(int argc, char** argv)
{
    Conv2dBlock<torch::nn::InstanceNorm2d, torch::nn::MaxPool2d> c_enc(
        std::vector<int64_t>{2, 10, 10},
        std::vector<int64_t>{16, 16},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1}, {2, 3}});

    torch::Tensor con_out_sz = c_enc->output_size();

    ConvTranspose2dBlock<torch::nn::InstanceNorm2d, torch::nn::MaxPool2d> c_dec(
        std::vector<int64_t>{con_out_sz[0].item<int>(), con_out_sz[1].item<int>(), con_out_sz[2].item<int>()},
        std::vector<int64_t>{16, 2},
        AnyModuleList{torch::nn::ReLU(), torch::nn::GELU()},
        std::vector<std::vector<int64_t>>{{1}, {2, 3}});

    MLPBlock mlp_enc(con_out_sz.prod().item<int>(), 2, std::vector<int64_t>{64, 64});

    MLPBlock mlp_dec(2, con_out_sz.prod().item<int>(), std::vector<int64_t>{64, 64});

    Quantizer quant(2, 100);
    quant->init_codebook_kmeans(-100, 100);

    torch::nn::Sequential encoder(c_enc, functional::flatten(1), mlp_enc);
    torch::nn::Sequential decoder(mlp_dec, functional::unflatten(1, con_out_sz), c_dec);

    VQVAE<Quantizer> model(encoder, decoder, quant);

    std::cout << model << std::endl;
    std::cout << model->centers() << std::endl;

    return 0;
}
