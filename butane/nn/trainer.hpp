#pragma once

#include "../data/dataloader.hpp"
#include "nn.hpp"
#include <boost/optional.hpp>

namespace nn {
    template <typename Model, typename Scheduler = torch::optim::LRScheduler>
    class ModelTrainer {
    public:
        ModelTrainer(
            Model& model,
            data::Dataloader& dl,
            std::shared_ptr<torch::optim::Optimizer> optimizer,
            boost::optional<std::shared_ptr<Scheduler>> scheduler = boost::none)
            : _model(model), _dl(dl), _optimizer(std::move(optimizer)), _scheduler(std::move(scheduler)) {}

        template <typename F>
        void operator()(unsigned int epochs, F loss, int eval_period = 0, boost::optional<data::Dataloader&> eval_dl = boost::none)
        {
            for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
                std::cout << "Epoch " << epoch << ": ";
                _model->step(_dl, _optimizer, loss, _scheduler);
                if (eval_period && eval_dl && !((epoch + 1) % eval_period))
                    eval(eval_dl.value(), loss);
            }
        }

        template <typename F>
        void eval(data::Dataloader& eval_dl, F loss)
        {
            _model->eval();
            std::cout << "Evaluation -> ";
            _model->step(eval_dl, _optimizer, loss, _scheduler);
            _model->train();
        }

    private:
        Model& _model;
        data::Dataloader& _dl;
        std::shared_ptr<torch::optim::Optimizer> _optimizer;
        boost::optional<std::shared_ptr<Scheduler>> _scheduler;
    };
} // namespace nn
