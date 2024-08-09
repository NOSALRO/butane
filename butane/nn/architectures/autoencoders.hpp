#pragma once

#include "../../data/dataloader.hpp"
#include "../nn.hpp"

namespace butane {
    namespace nn {
        template <typename Encoder, typename Decoder>
        class AEImpl : public torch::nn::Module {
        public:
            Encoder& encoder_;
            Decoder& decoder_;

            AEImpl(Encoder& encoder, Decoder& decoder) : encoder_(encoder), decoder_(decoder)
            {
                register_module("Encoder", encoder_);
                register_module("Decoder", decoder_);
            }

            torch::Tensor forward(torch::Tensor x)
            {
                torch::Tensor z = encoder_->forward(x);
                torch::Tensor reconstructed = decoder_->forward(z);
                return reconstructed;
            }

            torch::Tensor encode(torch::Tensor x)
            {
                return encoder_->forward(x);
            }

            torch::Tensor decode(torch::Tensor x)
            {
                return decoder_->forward(x);
            }

            torch::Device device()
            {
                return this->parameters()[0].device();
            }

            static torch::Tensor loss_fn(torch::Tensor x_hat, torch::Tensor x)
            {
                return torch::nn::functional::mse_loss(x_hat, x);
            }

            template <typename F, typename Scheduler = torch::optim::LRScheduler>
            void step(
                data::Dataloader& dl,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                F loss_fn_,
                std::shared_ptr<Scheduler> scheduler = nullptr)
            {
                float sum_loss = 0.f;
                int n_batches = dl.batches();

                for (auto batch : dl) {
                    this->is_training() ? optimizer->zero_grad() : noop;
                    torch::Tensor x_reconstructed = this->forward(batch.data);
                    torch::Tensor loss = loss_fn_(x_reconstructed, batch.data);
                    if (this->is_training()) {
                        loss.backward();
                        optimizer->step();
                    }

                    sum_loss += loss.item<float>();
                }

                if (this->is_training() && scheduler)
                    scheduler->step();

                float avg_loss = sum_loss / n_batches;
                std::cout << "Loss: " << avg_loss << std::endl;
            }
        };

        TORCH_MODULE_TEMPLATED(AE);

        // Deduction guide
        template <typename Encoder, typename Decoder>
        AE(Encoder, Decoder) -> AE<Encoder, Decoder>;

        template <typename Encoder, typename Decoder, typename Quantizer>
        class VQVAEImpl : public AEImpl<Encoder, Decoder> {
        public:
            Quantizer& quantizer_;

            VQVAEImpl(Encoder& encoder, Decoder& decoder, Quantizer& quantizer)
                : AEImpl<Encoder, Decoder>(encoder, decoder), quantizer_(quantizer)
            {
                this->register_module("Quantizer", quantizer_);
            }

            std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                torch::Tensor z = this->encoder_->forward(x);
                torch::Tensor quantized_z, quantization_loss;
                std::tie(quantized_z, quantization_loss) = quantizer_->forward(z);
                torch::Tensor reconstructed = this->decoder_->forward(quantized_z);
                return {reconstructed, quantization_loss};
            }

            torch::Tensor quantize(torch::Tensor x)
            {
                torch::Tensor z = this->encoder_->forward(x);
                return std::get<0>(quantizer_->forward(z));
            }

            torch::Tensor centers()
            {
                return quantizer_->centers();
            }

            static std::tuple<torch::Tensor, torch::Tensor> loss_fn(torch::Tensor x_hat, torch::Tensor x, torch::Tensor vq_loss)
            {
                torch::Tensor loss, x_rec;
                x_rec = torch::nn::functional::mse_loss(x_hat, x);
                loss = x_rec + vq_loss;
                return {loss, x_rec};
            }

            template <typename F, typename Scheduler = torch::optim::LRScheduler>
            void step(
                data::Dataloader& dl,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                F loss_fn_,
                std::shared_ptr<Scheduler> scheduler = nullptr)
            {
                float sum_loss = 0.f;
                float sum_quantization_loss = 0.f;
                float sum_loss_rec = 0.f;
                int n_batches = dl.batches();

                for (auto batch : dl) {
                    this->is_training() ? optimizer->zero_grad() : noop;
                    torch::Tensor x_reconstructed, quantization_loss;
                    std::tie(x_reconstructed, quantization_loss) = this->forward(batch.data);
                    torch::Tensor loss, loss_rec;
                    std::tie(loss, loss_rec) = loss_fn_(x_reconstructed, batch.data, quantization_loss);
                    if (this->is_training()) {
                        loss.backward();
                        optimizer->step();
                    }

                    sum_loss += loss.item<float>();
                    sum_quantization_loss += quantization_loss.item<float>();
                    sum_loss_rec += loss_rec.item<float>();
                }

                if (this->is_training() && scheduler)
                    scheduler->step();

                float avg_loss = sum_loss / n_batches;
                float avg_quantization_loss = sum_quantization_loss / n_batches;
                float avg_loss_rec = sum_loss_rec / n_batches;
                std::cout << "Loss: " << avg_loss << " Reconstruction Loss: " << avg_loss_rec << " Quantization Loss: " << avg_quantization_loss << std::endl;
            }
        };

        TORCH_MODULE_TEMPLATED(VQVAE);

        template <typename Encoder, typename Decoder, typename Quantizer>
        VQVAE(Encoder, Decoder, Quantizer) -> VQVAE<Encoder, Decoder, Quantizer>;

        template <typename Encoder, typename Decoder, typename Quantizer>
        class MLVQVAEImpl : public VQVAEImpl<Encoder, Decoder, Quantizer> {
        public:
            MLVQVAEImpl(Encoder& encoder, Decoder& decoder, Quantizer& quantizer) : VQVAEImpl<Encoder, Decoder, Quantizer>(encoder, decoder, quantizer) {}

            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x)
            {
                torch::Tensor z = this->encoder_->forward(x);
                torch::Tensor reconstructed = this->decoder_->forward(z);

                torch::Tensor quantized_z, quantization_loss;
                std::tie(quantized_z, quantization_loss) = this->quantizer_->forward(z);
                torch::Tensor quantized_reconstructed = this->decoder_->forward(quantized_z);
                return {reconstructed, quantized_reconstructed, quantization_loss};
            }

            static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> loss_fn(torch::Tensor x_hat, torch::Tensor x_quantized_hat, torch::Tensor x, torch::Tensor vq_loss)
            {
                torch::Tensor loss, x_rec, x_rec_quantized;
                x_rec = torch::nn::functional::mse_loss(x_hat, x);
                x_rec_quantized = torch::nn::functional::mse_loss(x_quantized_hat, x);
                loss = x_rec + x_rec_quantized + vq_loss;
                return {loss, x_rec, x_rec_quantized};
            }

            template <typename F, typename Scheduler = torch::optim::LRScheduler>
            void step(
                data::Dataloader& dl,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                F loss_fn_,
                std::shared_ptr<Scheduler> scheduler = nullptr)
            {
                float sum_loss = 0.f;
                float sum_quantization_loss = 0.f;
                float sum_loss_ae = 0.f;
                float sum_loss_vq = 0.f;
                int n_batches = dl.batches();

                for (auto batch : dl) {
                    this->is_training() ? optimizer->zero_grad() : noop;
                    torch::Tensor x_reconstructed, x_quantized_reconstructed, quantization_loss;
                    std::tie(x_reconstructed, x_quantized_reconstructed, quantization_loss) = this->forward(batch.data);
                    torch::Tensor loss, loss_ae, loss_vq;
                    std::tie(loss, loss_ae, loss_vq) = loss_fn_(x_reconstructed, x_quantized_reconstructed, batch.data, quantization_loss);

                    if (this->is_training()) {
                        loss.backward();
                        optimizer->step();
                    }

                    sum_loss += loss.item<float>();
                    sum_quantization_loss += quantization_loss.item<float>();
                    sum_loss_ae += loss_ae.item<float>();
                    sum_loss_vq += loss_vq.item<float>();
                }

                if (this->is_training() && scheduler)
                    scheduler->step();

                float avg_loss = sum_loss / n_batches;
                float avg_quantization_loss = sum_quantization_loss / n_batches;
                float avg_loss_ae = sum_loss_ae / n_batches;
                float avg_loss_vq = sum_loss_vq / n_batches;
                std::cout << "Loss: " << avg_loss << " Reconstruction Loss: " << avg_loss_ae << " Reconstruction Quant Loss: " << avg_loss_vq << " Quantization Loss: " << avg_quantization_loss << std::endl;
            }
        };
        TORCH_MODULE_TEMPLATED(MLVQVAE);

        template <typename Encoder, typename Decoder, typename Quantizer>
        MLVQVAE(Encoder, Decoder, Quantizer) -> MLVQVAE<Encoder, Decoder, Quantizer>;

        namespace utils {
            template <typename Model>
            struct does_clustering : std::true_type {};

            template <typename Encoder, typename Decoder>
            struct does_clustering<AE<Encoder, Decoder>> : std::false_type {};
        } // namespace utils
    } // namespace nn
} // namespace butane
