#pragma once

#include <algorithm>
#include <torch/torch.h>

namespace butane {
    namespace optim {
        class CyclicLR : public torch::optim::LRScheduler {
        public:
            CyclicLR(torch::optim::Optimizer& optimizer, double base_lr, double max_lr, int step_size)
                : torch::optim::LRScheduler(optimizer), base_lr_(base_lr), max_lr_(max_lr), step_size_(step_size)
            {
                total_size_ = 2 * step_size_;
                step_ratio_ = static_cast<double>(step_size_) / total_size_;
            }

            std::vector<double> get_lrs()
            {
                std::vector<double> params = get_current_lrs();
                for (size_t i = 0; i < params.size(); ++i) {
                    params[i] = calc_lr();
                }
                return params;
            }

        private:
            double calc_lr()
            {
                int cycle = std::floor(1.0 + static_cast<double>(step_count_) / total_size_);
                double x = 1.0 + static_cast<double>(step_count_) / total_size_ - cycle;

                double scale_factor;
                if (x <= step_ratio_) {
                    scale_factor = x / step_ratio_;
                }
                else {
                    scale_factor = (x - 1) / (step_ratio_ - 1);
                }

                double base_height = (max_lr_ - base_lr_) * scale_factor;
                return base_lr_ + base_height;
            }

            double base_lr_;
            double max_lr_;
            int step_size_;
            int total_size_;
            double step_ratio_;
        };
    } // namespace optim
} // namespace butane
