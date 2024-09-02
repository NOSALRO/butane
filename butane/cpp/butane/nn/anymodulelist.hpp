#pragma once

#include "nn.hpp"

namespace butane {
    namespace nn {
        class AnyModuleList {
        public:
            AnyModuleList() = default;

            template <typename... Modules>
            AnyModuleList(Modules&&... modules)
            {
                (_modules.push_back(torch::nn::AnyModule(std::forward<Modules>(modules))), ...);
            }

            template <typename Module>
            void push_back(Module module)
            {
                _modules.push_back(torch::nn::AnyModule(module));
            }

            std::vector<torch::nn::AnyModule> modules() { return _modules; }
            const size_t size() const { return _modules.size(); }

            torch::nn::AnyModule operator[](size_t index) const { return _modules[index]; }

            template <typename Module>
            void set(size_t index, Module module)
            {
                _modules[index] = torch::nn::AnyModule(module);
            }

            template <typename Module>
            void resize(size_t sz, Module module)
            {
                for (size_t i = _modules.size(); i < sz; ++i)
                    _modules.push_back(torch::nn::AnyModule(module));
            }

        private:
            std::vector<torch::nn::AnyModule> _modules;
        };
    } // namespace nn
} // namespace butane
