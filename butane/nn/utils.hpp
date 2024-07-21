#pragma once

#include "nn.hpp"

#define noop (void)0

#define TORCH_MODULE_IMPL_TEMPLATED(Name, ImplType)                     \
    template <typename... Opts>                                         \
    class Name : public torch::nn::ModuleHolder<ImplType<Opts...>> {    \
    public:                                                             \
        using torch::nn::ModuleHolder<ImplType<Opts...>>::ModuleHolder; \
        using Impl TORCH_UNUSED_EXCEPT_CUDA = ImplType<Opts...>;        \
    }

#define TORCH_MODULE_TEMPLATED(Name) TORCH_MODULE_IMPL_TEMPLATED(Name, Name##Impl)

namespace nn {

    namespace dtypes {
        using ivec = std::vector<int64_t>;
        using ivec2d = std::vector<std::vector<int64_t>>;

        using bvec = std::vector<bool>;
        using dvec = std::vector<double>;

        using conv_padding_mode_t = torch::nn::detail::conv_padding_mode_t;
        using pad_enum_vec = std::vector<conv_padding_mode_t>;

        template <int N>
        using ConvTranspose = typename std::conditional<N == 1, torch::nn::ConvTranspose1d,
            typename std::conditional<N == 2, torch::nn::ConvTranspose2d,
                torch::nn::ConvTranspose3d>::type>::type;

        template <int N>
        using Conv = typename std::conditional<N == 1, torch::nn::Conv1d,
            typename std::conditional<N == 2, torch::nn::Conv2d,
                torch::nn::Conv3d>::type>::type;
    }; // namespace dtypes

    class ModelBase {
    protected:
        template <typename T>
        bool vec_of_zeros(const std::vector<T>& v)
        {
            return std::all_of(v.begin(), v.end(), [](const T& i) { return i == 0; });
        }

        template <typename DefaultValue>
        void _fill_defaults(DefaultValue& parameter, size_t n_blocks)
        {
            if (parameter.size() == 1)
                parameter.resize(n_blocks, parameter[0]);
        }

        template <typename DefaultValue>
        void _fill_vec_defaults(std::vector<DefaultValue>& parameter, int N)
        {
            for (auto& i : parameter)
                if (i.size() == 1)
                    i = DefaultValue(N, i[0]);
        }
    };
} // namespace nn
