#include <torch/torch.h>
#include <butane/butane.h>

int main()
{
    torch::Tensor rdata = torch::empty({1e+6, 2}).uniform_(static_cast<double>(-1), static_cast<double>(1));
    rdata = rdata.to(torch::Device(torch::kCUDA));
    butane::clustring::MiniBatchKMeans kmeans(200, butane::KMeansPlusPlus, 1024, 500, 1e-4, 3);
    kmeans.fit(rdata);
    std::cout << kmeans.centroids() << std::endl;
    return 0;
}