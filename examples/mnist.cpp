#include "butane/butane.h"

using namespace nn;

int main(int argc, char** argv)
{
    torch::Device dev(torch::kCUDA);
    Conv2dBlockImpl<torch::nn::BatchNorm2d, torch::nn::MaxPool2d>::Config co{
        .input_dims = {1, 28, 28},
        .channels = {32, 32},
        .activation_function = {torch::nn::ReLU(), torch::nn::ReLU()},
        .conv_stride = {{1}, {1}},
        .pool_kernels = {{0}, {3}},
        .normalization = {false, true}};

    Conv2dBlock<torch::nn::BatchNorm2d, torch::nn::MaxPool2d> c(co);

    MLPBlock mlp(c->output_size().prod().item<int>(), 10, std::vector<int64_t>{64, 64}, AnyModuleList{torch::nn::ReLU(), torch::nn::ReLU(), torch::nn::LogSoftmax(-1)}, true);

    torch::nn::Sequential model(c, functional::flatten(1), mlp);
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
            auto label = batch.target.to(dev);
            // auto label = torch::zeros({16, 10}).scatter_(1, batch.target.unsqueeze(1), 1.0);
            torch::Tensor out = model->forward(data);
            torch::Tensor loss = torch::nn::functional::nll_loss(out, label);
            loss.backward();
            optimizer.step();
            sum_loss += loss.cpu();
            j += 1;
        }
        std::cout << "LOSS: " << sum_loss.item<double>() / j << std::endl;
        model->eval();
        int total_correct = 0;
        int total_samples = 0;
        for (auto& batch : *test_loader) {
            torch::Tensor out = model->forward(batch.data.to(dev));
            auto prediction = out.argmax(1).cpu();
            total_correct += prediction.eq(batch.target).sum().item<int>();
            total_samples += batch.target.size(0);
        }
        std::cout << "ACCURACY: -> " << static_cast<float>(total_correct) / total_samples << std::endl;
        model->train();
    }

    return 0;
}
