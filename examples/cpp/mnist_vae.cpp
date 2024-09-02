#include "butane/butane.h"

torch::Tensor loss_fn(torch::Tensor x_hat, torch::Tensor x)
{
    return torch::nn::functional::mse_loss(x_hat, x, torch::kSum);
}

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
        .normalization = {true, false}};

    butane::nn::Conv2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_enc(co_enc);

    butane::nn::ProbabilisticMLPBlock mlp_enc(
        c_enc->output_size().prod().item<int>(),
        100,
        std::vector<int64_t>{64, 64},
        butane::nn::AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::Tanh()},
        false);

    butane::nn::MLPBlock mlp_dec(100, c_enc->output_size().prod().item<int>(), std::vector<int64_t>{64, 64}, butane::nn::AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::ReLU()}, false);

    butane::nn::ConvTransposeBlockOptions co_dec{
        .input_dims = {32, 28, 28},
        .channels = {32, 1},
        .activation_function = {torch::nn::GELU(), torch::nn::Sigmoid()},
        .conv_stride = {{1}, {1}},
        .conv_bias = {true},
        .pool_kernels = {{0}, {0}},
        .output_activation = false,
        .normalization = {true, false}};

    butane::nn::ConvTranspose2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_dec(co_dec);

    torch::nn::Sequential encoder(c_enc, butane::nn::functional::flatten(1), mlp_enc);
    torch::nn::Sequential decoder(mlp_dec, butane::nn::functional::unflatten(1, c_enc->output_size()), c_dec);
    butane::nn::VAE model(encoder, decoder);
    model->set_beta(10.);
    model->reduction(butane::Sum);
    model->to(dev);
    std::cout << model << std::endl;

    butane::data::Dataset ds("data/mnist_data.pt", "data/mnist_targets.pt");
    butane::data::Dataset test_ds("data/mnist_test_data.pt", "data/mnist_test_targets.pt");
    ds.to(dev);
    test_ds.to(dev);
    butane::data::ops::drop(ds, 0.5);
    butane::data::Dataloader dl(ds, 64);
    butane::data::Dataloader test_dl(ds, 64, false);

    std::shared_ptr<torch::optim::Adam> optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions().lr(1e-03));
    std::shared_ptr<butane::optim::CyclicLR> scheduler = std::make_shared<butane::optim::CyclicLR>(*optimizer, 1e-04, 1e-03, 2000);

    butane::nn::ModelTrainer trainer(model, dl, optimizer);
    trainer(200, model->loss_fn, 20, test_dl);
    trainer.eval(test_dl, loss_fn);

    torch::save(model->encode(test_ds.data()), ".tmp/l.pt");
    torch::save(std::get<0>(model->forward(test_ds.data())), ".tmp/r.pt");
    // torch::save(model->forward(test_ds.data()), ".tmp/r.pt");
    torch::save(test_ds.data(), ".tmp/o.pt");

    return 0;
}
