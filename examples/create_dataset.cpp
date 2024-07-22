#include "butane/butane.h"

int main(int argc, char** argv)
{
    torch::Device dev(torch::kCUDA);
    auto train_dataset = torch::data::datasets::MNIST("./data/").map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), 128);

    data::Dataset ds;

    int iters = 0;
    for (auto& batch : *data_loader) {
        ds.append(batch.data, batch.target);
        iters++;
        if (iters == 5)
            break;
        auto a = ds.get(2);
        // if (a.second.numel()) {
        //     std::cout << a.first << " " << a.second << std::endl;
        // }
    }
    // auto ds1 = data::Dataset(ds.data(), ds.targets()).map(torch::data::transforms::Stack<>());
    auto ds1 = torch::data::datasets::map(ds, torch::data::transforms::Stack<>());
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(ds1), 128);

    for (auto& batch : *dl) {
        auto d = batch;
        std::cout << d.target.sizes() << std::endl;
    }

    return 0;
}
