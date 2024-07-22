#include "butane/butane.h"

int main(int argc, char** argv)
{
    torch::Device dev(torch::kCUDA);
    nn::Conv2dBlockImpl<torch::nn::BatchNorm2d, torch::nn::MaxPool2d>::Config co_enc{
        .input_dims = {1, 28, 28},
        .channels = {32, 32},
        .activation_function = {torch::nn::GELU(), torch::nn::GELU()},
        .conv_stride = {{1}, {1}},
        .conv_bias = {true},
        .pool_kernels = {{0}, {0}},
        .normalization = {true, true}};

    nn::Conv2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_enc(co_enc);

    nn::MLPBlock mlp_enc(c_enc->output_size().prod().item<int>(), 100, std::vector<int64_t>{64, 64}, nn::AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::Tanh()}, false);

    nn::Quantizer quantizer(100, 10, dev);
    quantizer->set_beta(1.25);
    quantizer->init_codebook_kmeans(-.3, .3);

    nn::MLPBlock mlp_dec(100, c_enc->output_size().prod().item<int>(), std::vector<int64_t>{64, 64}, nn::AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::ReLU()}, false);

    nn::ConvTranspose2dBlockImpl<torch::nn::BatchNorm2d, torch::nn::MaxPool2d>::Config co_dec{
        .input_dims = {32, 28, 28},
        .channels = {32, 1},
        .activation_function = {torch::nn::GELU(), torch::nn::Sigmoid()},
        .conv_stride = {{1}, {1}},
        .conv_bias = {true},
        .pool_kernels = {{0}, {0}},
        .output_activation = false,
        .normalization = {false, false}};

    nn::ConvTranspose2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c_dec(co_dec);


    torch::nn::Sequential encoder(c_enc, nn::functional::flatten(1), mlp_enc);
    torch::nn::Sequential decoder(mlp_dec, nn::functional::unflatten(1, c_enc->output_size()), c_dec);
    nn::MLVQVAE<nn::Quantizer> model(encoder, decoder, quantizer);
    model->to(dev);
    std::cout << model << std::endl;

    auto train_dataset = torch::data::datasets::MNIST("./data/")
                             .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), 128);

    auto test_dataset = torch::data::datasets::MNIST("./data/", torch::data::datasets::MNIST::Mode::kTest)
                            .map(torch::data::transforms::Stack<>());
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), 64);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions().lr(1e-04));

    for (int i = 0; i < 100; i++) {
        torch::Tensor sum_loss = torch::tensor(0.);
        int j = 0;
        for (auto& batch : *data_loader) {
            optimizer.zero_grad();
            auto data = batch.data.to(dev);
            auto [out, out_q, quant_loss] = model->forward(data);
            torch::Tensor loss = torch::nn::functional::mse_loss(out, data) + quant_loss + torch::nn::functional::mse_loss(out_q, out);
            loss.backward();
            optimizer.step();
            sum_loss += loss.cpu();
            j += 1;
        }
        std::cout << "LOSS: " << sum_loss.item<double>() / j << std::endl;
        model->eval();
        float test_loss = 0;
        int total_samples = 0;
        for (auto& batch : *test_loader) {
            auto [out, out_q, l] = model->forward(batch.data.to(dev));
            torch::save(batch.data, ".tmp/o.pt");
            torch::save(out_q, ".tmp/r.pt");
            test_loss += torch::nn::functional::mse_loss(out, batch.data.to(dev)).cpu().item<float>();
            total_samples += batch.data.size(0);
        }
        std::cout << "TEST_LOSS: -> " << static_cast<float>(test_loss) / total_samples << std::endl;
        model->train();
    }

    return 0;
}
