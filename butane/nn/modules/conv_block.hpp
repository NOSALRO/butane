#pragma once

#include "../nn.hpp"

namespace butane {
    namespace nn {
        using ivec = dtypes::ivec;
        using ivec2d = dtypes::ivec2d;
        using bvec = dtypes::bvec;
        using dvec = dtypes::dvec;
        using conv_padding_mode_t = dtypes::conv_padding_mode_t;
        using pad_enum_vec = dtypes::pad_enum_vec;

        struct ConvBlockOptions {
            ivec input_dims;
            ivec channels;
            AnyModuleList activation_function = AnyModuleList(torch::nn::ReLU());
            ivec2d conv_kernels = ivec2d{{3}};
            ivec2d conv_stride = ivec2d{{1}};
            ivec2d conv_pad = ivec2d{{1}};
            bvec conv_bias = bvec{false};
            pad_enum_vec conv_pad_mode = pad_enum_vec{torch::kZeros};
            ivec2d pool_kernels = ivec2d{{0}};
            ivec2d pool_stride = ivec2d{{1}};
            ivec2d pool_pad = ivec2d{{1}};
            dvec dropout = dvec{0.0};
            bool output_activation = true;
            bvec normalization = bvec{false};
        };

        struct ConvTransposeBlockOptions {
            ivec input_dims;
            ivec channels;
            AnyModuleList activation_function = AnyModuleList(torch::nn::ReLU());
            ivec2d conv_kernels = ivec2d{{3}};
            ivec2d conv_stride = ivec2d{{1}};
            ivec2d conv_pad = ivec2d{{1}};
            bvec conv_bias = bvec{false};
            ivec2d conv_output_pad = ivec2d{{0}};
            ivec2d pool_kernels = ivec2d{{0}};
            ivec2d pool_stride = ivec2d{{1}};
            ivec2d pool_pad = ivec2d{{1}};
            dvec dropout = dvec{0.0};
            bool output_activation = true;
            bvec normalization = bvec{false};
        };

        template <int N, typename Norm, typename Pool>
        class ConvNdBlockImpl : public torch::nn::Module {
        public:
            using Conv = dtypes::Conv<N>;
            ConvNdBlockImpl() = default;

            ConvNdBlockImpl(const ConvBlockOptions& conf)
            {
                *this = ConvNdBlockImpl(
                    conf.input_dims, conf.channels,
                    conf.activation_function, conf.conv_kernels,
                    conf.conv_stride, conf.conv_pad,
                    conf.conv_bias, conf.conv_pad_mode,
                    conf.pool_kernels, conf.pool_stride,
                    conf.pool_pad, conf.dropout,
                    conf.output_activation, conf.normalization);
            }

            ConvNdBlockImpl(
                const ivec& input_dims,
                const ivec& channels,
                AnyModuleList activation_function = AnyModuleList(torch::nn::ReLU()),
                ivec2d conv_kernels = ivec2d{{3}},
                ivec2d conv_stride = ivec2d{{1}},
                ivec2d conv_pad = ivec2d{{1}},
                bvec conv_bias = bvec{false},
                pad_enum_vec conv_pad_mode = pad_enum_vec{torch::kZeros},
                ivec2d pool_kernels = ivec2d{{0}},
                ivec2d pool_stride = ivec2d{{1}},
                ivec2d pool_pad = ivec2d{{1}},
                dvec dropout = dvec{0.0},
                bool output_activation = false,
                bvec normalization = bvec{false}) : _input_dims(input_dims), _channels(channels)
            {

                size_t n_blocks = _channels.size();

                assert((activation_function.size() == n_blocks || activation_function.size() == 1) && "Activation Functions size should be 1 or N_Channels");
                assert((conv_kernels.size() == n_blocks || conv_kernels.size() == 1) && "Conv kernels size should be 1 or N_Channels");
                assert((conv_stride.size() == n_blocks || conv_stride.size() == 1) && "Conv stride size should be 1 or N_Channels");
                assert((conv_pad.size() == n_blocks || conv_pad.size() == 1) && "Conv pad size should be 1 or N_Channels");
                assert((conv_pad_mode.size() == n_blocks || conv_pad_mode.size() == 1) && "Conv padding mode size should be 1 or N_Channels");
                assert((conv_bias.size() == n_blocks || conv_bias.size() == 1) && "Conv bias size should be 1 or N_Channels");
                assert((pool_kernels.size() == n_blocks || pool_kernels.size() == 1) && "Pool kernels size should be 1 or N_Channels");
                assert((pool_stride.size() == n_blocks || pool_stride.size() == 1) && "Pool stride size should be 1 or N_Channels");
                assert((pool_pad.size() == n_blocks || pool_pad.size() == 1) && "Pool pad size should be 1 or N_Channels");
                assert((dropout.size() == n_blocks || dropout.size() == 1) && "Dropout size should be 1 or N_Channels");
                assert((normalization.size() == n_blocks || normalization.size() == 1) && "Normalization size should be 1 or N_Channels");

                utils::_fill_vec_defaults(conv_kernels, N);
                utils::_fill_vec_defaults(conv_stride, N);
                utils::_fill_vec_defaults(conv_pad, N);
                utils::_fill_vec_defaults(pool_kernels, N);
                utils::_fill_vec_defaults(pool_stride, N);
                utils::_fill_vec_defaults(pool_pad, N);

                utils::_fill_defaults(activation_function, n_blocks);
                utils::_fill_defaults(conv_kernels, n_blocks);
                utils::_fill_defaults(conv_stride, n_blocks);
                utils::_fill_defaults(conv_pad, n_blocks);
                utils::_fill_defaults(conv_pad_mode, n_blocks);
                utils::_fill_defaults(conv_bias, n_blocks);
                utils::_fill_defaults(pool_kernels, n_blocks);
                utils::_fill_defaults(pool_stride, n_blocks);
                utils::_fill_defaults(pool_pad, n_blocks);
                utils::_fill_defaults(normalization, n_blocks);
                utils::_fill_defaults(dropout, n_blocks);

                if (!output_activation) {
                    activation_function.set(activation_function.size() - 1, torch::nn::Identity());
                }

                _channels.insert(_channels.begin(), _input_dims[0]);
                for (unsigned int i = 0; i < _channels.size() - 1; ++i)
                    _create_subblock(
                        _seq,
                        _channels[i],
                        _channels[i + 1],
                        conv_kernels[i],
                        conv_stride[i],
                        conv_pad[i],
                        conv_bias[i],
                        conv_pad_mode[i],
                        pool_kernels[i],
                        pool_stride[i],
                        pool_pad[i],
                        dropout[i],
                        activation_function[i],
                        normalization[i]);

                register_module("Conv" + std::to_string(N) + "d", _seq);
            }

            torch::Tensor forward(torch::Tensor x)
            {
                return _seq->forward(x);
            }

            torch::Tensor output_size()
            {
                torch::NoGradGuard no_grad;
                this->eval();
                torch::Tensor sz = torch::tensor(this->forward(torch::rand(_input_dims).unsqueeze(0)).squeeze(0).sizes());
                this->train();
                return sz;
            }

            const int out_features() const { return _channels[_channels.size() - 1]; }

        private:
            ivec _input_dims, _channels;
            torch::nn::Sequential _seq;

            void _create_subblock(
                torch::nn::Sequential& seq,
                int in_channels,
                int out_channels,
                ivec conv_kernel,
                ivec conv_stride,
                ivec conv_pad,
                bool conv_bias,
                conv_padding_mode_t conv_pad_mode,
                ivec pool_kernel,
                ivec pool_stride,
                ivec pool_pad,
                double dropout,
                torch::nn::AnyModule af,
                bool normalization)
            {
                torch::nn::ConvOptions<N> conv_opts = torch::nn::ConvOptions<N>(in_channels, out_channels, conv_kernel).stride(conv_stride).padding(conv_pad).bias(conv_bias).padding_mode(conv_pad_mode);
                _seq->push_back(Conv(conv_opts));

                normalization ? _seq->push_back(Norm(out_channels)) : noop;
                _seq->push_back(af);

                if (!utils::_vec_of_zeros(pool_kernel)) {
                    auto pool_opts = Pool(0)->options;
                    pool_opts.kernel_size(pool_kernel).stride(pool_stride);//.padding(pool_pad);
                    _seq->push_back(Pool(pool_opts));
                }

                if (!(dropout == 0)) {
                    _seq->push_back(torch::nn::Dropout(dropout));
                }
            }
        };

        template <typename... Opts>
        using Conv1dBlockImpl = ConvNdBlockImpl<1, Opts...>;

        template <typename... Opts>
        using Conv2dBlockImpl = ConvNdBlockImpl<2, Opts...>;

        template <typename... Opts>
        using Conv3dBlockImpl = ConvNdBlockImpl<3, Opts...>;

        TORCH_MODULE_TEMPLATED(Conv1dBlock);
        TORCH_MODULE_TEMPLATED(Conv2dBlock);
        TORCH_MODULE_TEMPLATED(Conv3dBlock);

        template <int N, typename Norm, typename Pool>
        class ConvTransposeNdBlockImpl : public torch::nn::Module {
        public:
            using ConvTranspose = dtypes::ConvTranspose<N>;
            ConvTransposeNdBlockImpl() = default;

            ConvTransposeNdBlockImpl(const ConvTransposeBlockOptions& conf)
            {
                *this = ConvTransposeNdBlockImpl(
                    conf.input_dims, conf.channels,
                    conf.activation_function, conf.conv_kernels,
                    conf.conv_stride, conf.conv_pad,
                    conf.conv_bias, conf.conv_output_pad,
                    conf.pool_kernels, conf.pool_stride,
                    conf.pool_pad, conf.dropout,
                    conf.output_activation, conf.normalization);
            }

            ConvTransposeNdBlockImpl(
                const ivec& input_dims,
                const ivec& channels,
                AnyModuleList activation_function = AnyModuleList(torch::nn::ReLU()),
                ivec2d conv_kernels = ivec2d{{3}},
                ivec2d conv_stride = ivec2d{{1}},
                ivec2d conv_pad = ivec2d{{1}},
                bvec conv_bias = bvec{true},
                ivec2d conv_output_pad = ivec2d{{0}},
                ivec2d pool_kernels = ivec2d{{0}},
                ivec2d pool_stride = ivec2d{{1}},
                ivec2d pool_pad = ivec2d{{1}},
                dvec dropout = dvec{0.0},
                bool output_activation = false,
                bvec normalization = bvec{false}) : _input_dims(input_dims), _channels(channels)
            {

                size_t n_blocks = _channels.size();

                assert((activation_function.size() == n_blocks || activation_function.size() == 1) && "Activation Functions size should be 1 or N_Channels");
                assert((conv_kernels.size() == n_blocks || conv_kernels.size() == 1) && "Conv kernels size should be 1 or N_Channels");
                assert((conv_stride.size() == n_blocks || conv_stride.size() == 1) && "Conv stride size should be 1 or N_Channels");
                assert((conv_pad.size() == n_blocks || conv_pad.size() == 1) && "Conv pad size should be 1 or N_Channels");
                assert((conv_output_pad.size() == n_blocks || conv_output_pad.size() == 1) && "Conv output padding size should be 1 or N_Channels");
                assert((conv_bias.size() == n_blocks || conv_bias.size() == 1) && "Conv bias size should be 1 or N_Channels");
                assert((pool_kernels.size() == n_blocks || pool_kernels.size() == 1) && "Pool kernels size should be 1 or N_Channels");
                assert((pool_stride.size() == n_blocks || pool_stride.size() == 1) && "Pool stride size should be 1 or N_Channels");
                assert((pool_pad.size() == n_blocks || pool_pad.size() == 1) && "Pool pad size should be 1 or N_Channels");
                assert((dropout.size() == n_blocks || dropout.size() == 1) && "Dropout size should be 1 or N_Channels");
                assert((normalization.size() == n_blocks || normalization.size() == 1) && "Normalization size should be 1 or N_Channels");

                utils::_fill_vec_defaults(conv_kernels, N);
                utils::_fill_vec_defaults(conv_stride, N);
                utils::_fill_vec_defaults(conv_pad, N);
                utils::_fill_vec_defaults(conv_output_pad, N);
                utils::_fill_vec_defaults(pool_kernels, N);
                utils::_fill_vec_defaults(pool_stride, N);
                utils::_fill_vec_defaults(pool_pad, N);

                utils::_fill_defaults(activation_function, n_blocks);
                utils::_fill_defaults(conv_kernels, n_blocks);
                utils::_fill_defaults(conv_stride, n_blocks);
                utils::_fill_defaults(conv_pad, n_blocks);
                utils::_fill_defaults(conv_output_pad, n_blocks);
                utils::_fill_defaults(conv_bias, n_blocks);
                utils::_fill_defaults(pool_kernels, n_blocks);
                utils::_fill_defaults(pool_stride, n_blocks);
                utils::_fill_defaults(pool_pad, n_blocks);
                utils::_fill_defaults(normalization, n_blocks);
                utils::_fill_defaults(dropout, n_blocks);

                if (!output_activation) {
                    activation_function.set(activation_function.size() - 1, torch::nn::Identity());
                }

                _channels.insert(_channels.begin(), _input_dims[0]);
                for (unsigned int i = 0; i < _channels.size() - 1; ++i)
                    _create_subblock(
                        _seq,
                        _channels[i],
                        _channels[i + 1],
                        conv_kernels[i],
                        conv_stride[i],
                        conv_pad[i],
                        conv_bias[i],
                        conv_output_pad[i],
                        pool_kernels[i],
                        pool_stride[i],
                        pool_pad[i],
                        dropout[i],
                        activation_function[i],
                        normalization[i]);

                register_module("ConvTranspose" + std::to_string(N) + "d", _seq);
            }

            torch::Tensor forward(torch::Tensor x)
            {
                return _seq->forward(x);
            }

            torch::Tensor output_size()
            {
                torch::NoGradGuard no_grad;
                this->eval();
                torch::Tensor sz = torch::tensor(this->forward(torch::rand(_input_dims).unsqueeze(0)).squeeze(0).sizes());
                this->train();
                return sz;
            }

            torch::nn::Sequential _seq;

        private:
            ivec _input_dims, _channels;

            void _create_subblock(
                torch::nn::Sequential& seq,
                int in_channels,
                int out_channels,
                ivec conv_kernel,
                ivec conv_stride,
                ivec conv_pad,
                bool conv_bias,
                ivec conv_output_pad,
                ivec pool_kernel,
                ivec pool_stride,
                ivec pool_pad,
                double dropout,
                torch::nn::AnyModule af,
                bool normalization)
            {
                torch::nn::ConvTransposeOptions<N> conv_opts = torch::nn::ConvTransposeOptions<N>(in_channels, out_channels, conv_kernel).stride(conv_stride).padding(conv_pad).bias(conv_bias).output_padding(conv_output_pad);
                _seq->push_back(ConvTranspose(conv_opts));

                normalization ? _seq->push_back(Norm(out_channels)) : noop;
                _seq->push_back(af);

                if (!utils::_vec_of_zeros(pool_kernel)) {
                    auto pool_opts = Pool(0)->options;
                    pool_opts.kernel_size(pool_kernel).stride(pool_stride);//.padding(pool_pad);
                    _seq->push_back(Pool(pool_opts));
                }

                if (!(dropout == 0)) {
                    _seq->push_back(torch::nn::Dropout(dropout));
                }
            }
        };

        template <typename... Opts>
        using ConvTranspose1dBlockImpl = ConvTransposeNdBlockImpl<1, Opts...>;

        template <typename... Opts>
        using ConvTranspose2dBlockImpl = ConvTransposeNdBlockImpl<2, Opts...>;

        template <typename... Opts>
        using ConvTranspose3dBlockImpl = ConvTransposeNdBlockImpl<3, Opts...>;

        TORCH_MODULE_TEMPLATED(ConvTranspose1dBlock);
        TORCH_MODULE_TEMPLATED(ConvTranspose2dBlock);
        TORCH_MODULE_TEMPLATED(ConvTranspose3dBlock);
    } // namespace nn
} // namespace butane
