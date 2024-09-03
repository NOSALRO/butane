#include "butane/butane.h"

int main(int argc, char** argv)
{
    torch::Device dev(torch::kCUDA);
    butane::nn::ConvBlockOptions co_enc{
        .input_dims = {1, 28, 28},
        .channels = {32, 32},
        .activation_function = {torch::nn::GELU(), torch::nn::GELU()},
        .conv_stride = {{1}, {1}},
        .conv_bias = {true},
        .pool_kernels = {{0}, {0}},
        .normalization = {true, true}};

    butane::nn::Conv2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_enc(co_enc);

    butane::nn::MLPBlock mlp_enc(c_enc->output_size().prod().item<int>(), 100, std::vector<int64_t>{64, 64}, butane::nn::AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::Tanh()}, false);

    butane::nn::Quantizer quantizer(100, 10, dev);
    // butane::nn::STEQuantizer<torch::optim::AdamW> quantizer(100, 10, dev, 1.5, 2.0, 1);
    quantizer->set_beta(1.25);
    quantizer->init_codebook_kmeans(-.3, .3);

    butane::nn::MLPBlock mlp_dec(100, c_enc->output_size().prod().item<int>(), std::vector<int64_t>{64, 64}, butane::nn::AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::ReLU()}, false);

    butane::nn::ConvTransposeBlockOptions co_dec{
        .input_dims = {32, 28, 28},
        .channels = {32, 1},
        .activation_function = {torch::nn::GELU(), torch::nn::Sigmoid()},
        .conv_stride = {{1}, {1}},
        .conv_bias = {true},
        .pool_kernels = {{0}, {0}},
        .output_activation = false,
        .normalization = {false, false}};

    butane::nn::ConvTranspose2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_dec(co_dec);


    torch::nn::Sequential encoder(c_enc, butane::nn::functional::flatten(1), mlp_enc);
    torch::nn::Sequential decoder(mlp_dec, butane::nn::functional::unflatten(1, c_enc->output_size()), c_dec);
    butane::nn::MLVQVAE model(encoder, decoder, quantizer);
    model->to(dev);

    butane::data::Dataset ds("data/mnist_data.pt", "data/mnist_targets.pt");
    butane::data::Dataset test_ds("data/mnist_test_data.pt", "data/mnist_test_targets.pt");
    ds.to(dev);
    test_ds.to(dev);
    butane::data::ops::drop(ds, 0.5);
    butane::data::Dataloader dl(ds, 64);
    butane::data::Dataloader test_dl(ds, 64, false);

    std::shared_ptr<torch::optim::Adam> optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions().lr(1e-04));
    std::shared_ptr<butane::optim::CyclicLR> scheduler = std::make_shared<butane::optim::CyclicLR>(*optimizer, 1e-04, 1e-03, 2000);

    butane::nn::ModelTrainer trainer(model, dl, optimizer, scheduler);
    trainer(4, model->loss_fn, 3, test_dl);
    trainer.eval(test_dl, model->loss_fn);

    for (const auto& i : optimizer->param_groups())
        std::cout << i.options().get_lr() << std::endl;

    optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions().lr(1e-04));
    scheduler = std::make_shared<butane::optim::CyclicLR>(*optimizer, 1e-04, 1e-03, 2000);

    for (const auto& i : optimizer->param_groups())
        std::cout << i.options().get_lr() << std::endl;
    return 0;
}
