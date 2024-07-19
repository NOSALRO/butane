#pragma once
#include <torch/torch.h>

#include <utility>
#include <vector>

class AnyModuleList {
public:
    AnyModuleList() {}

    template <typename... Modules>
    AnyModuleList(Modules&&... modules)
    {
        (modules_.push_back(torch::nn::AnyModule(std::forward<Modules>(modules))), ...);
    }

    template <typename Module>
    void push_back(Module module)
    {
        modules_.push_back(torch::nn::AnyModule(module));
    }

    std::vector<torch::nn::AnyModule> modules() { return modules_; }
    const size_t size() const { return modules_.size(); }

    torch::nn::AnyModule operator[](size_t index) const { return modules_[index]; }

    template <typename Module>
    void set(size_t index, Module module) { 
        modules_[index] = torch::nn::AnyModule(module); 
    }

    template <typename Module>
    void resize(size_t sz, Module module) { 
        for (size_t i = modules_.size(); i < sz; ++i) 
            modules_.push_back(torch::nn::AnyModule(module));
    }

private:
    std::vector<torch::nn::AnyModule> modules_;
};
