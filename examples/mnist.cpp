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

    SimpleClassifier model(torch::nn::Sequential{c, functional::flatten(1), mlp});
    model->to(dev);
    std::cout << model << std::endl;

    data::Dataset ds("data/mnist_data.pt", "data/mnist_targets.pt");
    data::Dataset test_ds("data/mnist_test_data.pt", "data/mnist_test_targets.pt");
    ds.to(dev);
    test_ds.to(dev);
    data::Dataloader dl(ds, 64);
    data::Dataloader test_dl(ds, 64, false);

    std::shared_ptr<torch::optim::Adam> optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions().lr(1e-04));
    std::shared_ptr<optim::CyclicLR> scheduler = std::make_shared<optim::CyclicLR>(*optimizer, 1e-04, 1e-03, 2000);
    ModelTrainer<SimpleClassifier, optim::CyclicLR> trainer(model, dl, optimizer, scheduler);
    trainer(10, model->loss_fn, 3, test_dl);
    trainer.eval(test_dl, model->loss_fn);

    return 0;
}
