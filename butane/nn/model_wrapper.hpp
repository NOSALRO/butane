#pragma once

#include "../data/dataloader.hpp"
#include "nn.hpp"

namespace butane {
    namespace nn {
        class SimpleClassifierImpl : public torch::nn::Module {
        public:
            SimpleClassifierImpl(torch::nn::Sequential model) : _model(std::move(model))
            {
                register_module("Classifier", _model);
            }

            torch::Tensor forward(torch::Tensor x) { return _model->forward(x); }

            static torch::Tensor loss_fn(torch::Tensor y_hat, torch::Tensor y)
            {
                return torch::nn::functional::nll_loss(y_hat, y);
            }

            template <typename F, typename Scheduler = torch::optim::LRScheduler>
            void step(
                data::Dataloader& dl,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                F loss_fn_,
                boost::optional<std::shared_ptr<Scheduler>> scheduler = boost::none)
            {
                float sum_loss = 0.f;
                int64_t correct_preds = 0;
                int n_batches = dl.batches();

                for (auto batch : dl) {
                    _model->is_training() ? optimizer->zero_grad() : noop;
                    torch::Tensor y_hat = this->forward(batch.data);
                    torch::Tensor loss = loss_fn_(y_hat, batch.target);
                    correct_preds += y_hat.argmax(-1).eq(batch.target).sum().cpu().item<int64_t>();
                    if (_model->is_training()) {
                        loss.backward();
                        optimizer->step();
                    }
                    sum_loss += loss.item<float>();
                }

                if (_model->is_training() && scheduler)
                    scheduler.value()->step();

                float avg_loss = sum_loss / n_batches;
                float accuracy = correct_preds / static_cast<float>(dl.dataset().sizes()[0]);
                std::cout << "Loss: " << avg_loss << " Accuracy: " << accuracy << std::endl;
            }

        private:
            torch::nn::Sequential _model;
        };

        TORCH_MODULE(SimpleClassifier);
    } // namespace nn
} // namespace butane
