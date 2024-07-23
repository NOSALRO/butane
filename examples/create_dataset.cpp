#include <typeinfo>
#include "butane/butane.h"
#include "butane/data/dataloader.hpp"

int main(int argc, char** argv)
{
    torch::Device dev(torch::kCUDA);

    data::Dataset ds("data/mnist_data.pt", "data/mnist_targets.pt");
    std::cout << ds.data().sizes()  << " " << ds.targets().sizes() << std::endl;

    ds.to(dev);
    ds.drop_to_max_size(30000);

    std::cout << ds.get(torch::arange(3)).data.sizes() << std::endl;
    std::cout << ds.get(torch::arange(3)).target.sizes() << std::endl;
    data::Dataloader dl(ds, 64);
    std::cout << dl.batches() << std::endl;
    for (auto i : dl)
        std::cout << i.data.sizes() << std::endl;

    return 0;
}
