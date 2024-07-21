#pragma once

#include "modules.hpp"
#include <boost/optional.hpp>

namespace nn {
    template <typename Model>
    class ModelTrainer {
    public:
        ModelTrainer(Model& model, std::shared_ptr<torch::optim::Optimizer> optimizer, boost::optional<std::shared_ptr<torch::optim::LRScheduler>> scheduler = boost::none)
            : _model(model), _optimizer(optimizer), _scheduler(scheduler) {}

        void operator()(unsigned int epochs)
        {
            for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
                _model->step(_optimizer, _scheduler);
            }
        }

    private:
        Model& _model;
        std::shared_ptr<torch::optim::Optimizer> _optimizer;
        boost::optional<std::shared_ptr<torch::optim::LRScheduler>> _scheduler;
    };
} // namespace nn
