#include "butane/butane.h"

using namespace nn;

int main(int argc, char** argv)
{
    torch::Device dev(torch::kCUDA);
    Conv2dBlockImpl<torch::nn::BatchNorm2d, torch::nn::MaxPool2d>::Config co_enc{
        .input_dims = {1, 28, 28},
        .channels = {32, 32},
        .activation_function = {torch::nn::GELU(), torch::nn::GELU()},
        .conv_stride = {{1}, {1}},
        .pool_kernels = {{0}, {0}},
        .normalization = {true, true}};

    Conv2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_enc(co_enc);

    MLPBlock mlp_enc(c_enc->output_size().prod().item<int>(), 100, std::vector<int64_t>{64, 64}, AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::ReLU()}, false);

    Quantizer quantizer(100, 10, dev);
    quantizer->set_beta(1.25);

    MLPBlock mlp_dec(100, c_enc->output_size().prod().item<int>(), std::vector<int64_t>{64, 64}, AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::ReLU()}, false);

    ConvTranspose2dBlockImpl<torch::nn::BatchNorm2d, torch::nn::MaxPool2d>::Config co_dec{
        .input_dims = {32, 28, 28},
        .channels = {32, 1},
        .activation_function = {torch::nn::GELU(), torch::nn::Sigmoid()},
        .conv_stride = {{1}, {1}},
        .pool_kernels = {{0}, {0}},
        .output_activation = false,
        .normalization = {false, false}};

    ConvTranspose2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_dec(co_dec);


    torch::nn::Sequential encoder(c_enc, functional::flatten(1), mlp_enc);
    torch::nn::Sequential decoder(mlp_dec, functional::unflatten(1, c_enc->output_size()), c_dec);
    MLVQVAE<Quantizer> model(encoder, decoder, quantizer);
    model->to(dev);
    std::shared_ptr<torch::optim::Adam> optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions().lr(1e-04));
    ModelTrainer trainer(model, optimizer);
    trainer(100);

    return 0;
}
