#include "butane/butane.h"

class TMImpl : public torch::nn::Module {
public:
    torch::nn::Sequential seq;
    TMImpl()
    {
        seq = torch::nn::Sequential(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        return seq->forward(x);
    }
};
TORCH_MODULE(TM);

int main()
{
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

#if 0
    torch::Tensor rdata = torch::rand({100, 10});
    butane::data::Dataset ds, ds2;
    {
        auto t1 = high_resolution_clock::now();
        ds.append(rdata);
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
    }
    {
        auto t1 = high_resolution_clock::now();
        ds2.move(rdata);
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        std::cout << ms_double.count() << "ms\n";
    }
    std::cout << ds.data() << std::endl;
    std::cout << ds2.data() << std::endl;

    torch::Tensor rdata = -torch::rand({10, 2});
    TM l;
    auto t1 = high_resolution_clock::now();

    auto out = l->forward(rdata);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    std::cout << rdata << std::endl;
    std::cout << out << std::endl;
    std::cout << rdata << std::endl;
#endif

    butane::data::Dataset ds(-torch::rand({10, 2}));
    butane::data::Dataloader dl(ds, 1);
    TM l;

    for (auto batch : dl) {
        l->forward(batch.data);
    }
    std::cout << ds.data() << std::endl;

    return 0;
}
