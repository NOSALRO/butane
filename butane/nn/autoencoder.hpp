#pragma once

#include "nn.hpp"

namespace nn {
    class AEImpl : public torch::nn::Module {
    public:
        AEImpl(torch::nn::Sequential encoder, torch::nn::Sequential decoder) : _enc(std::move(encoder)), _dec(std::move(decoder))
        {
            register_module("Enc", _enc);
            register_module("Dec", _dec);
        }

        torch::Tensor forward(torch::Tensor x)
        {
            torch::Tensor z = _enc->forward(x);
            torch::Tensor reconstructed = _dec->forward(z);
            return reconstructed;
        }

        torch::Tensor encode(torch::Tensor x)
        {
            return _enc->forward(x);
        }

        torch::Tensor decode(torch::Tensor x)
        {
            return _dec->forward(x);
        }

    private:
        torch::nn::Sequential _enc, _dec;
    };

    TORCH_MODULE(AE);

    template <typename Quantizer>
    class VQVAEImpl : public torch::nn::Module {
    public:
        VQVAEImpl(torch::nn::Sequential encoder, torch::nn::Sequential decoder, Quantizer quantizer) : _enc(std::move(encoder)), _dec(std::move(decoder)), _quantizer(std::move(quantizer))
        {
            register_module("Enc", _enc);
            register_module("Quantizer", _quantizer);
            register_module("Dec", _dec);
        }

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
        {
            torch::Tensor z = _enc->forward(x);
            torch::Tensor quantized_z, quantization_loss;
            std::tie(quantized_z, quantization_loss) = _quantizer->forward(z);
            torch::Tensor reconstructed = _dec->forward(quantized_z);
            return {reconstructed, quantization_loss};
        }

        torch::Tensor encode(torch::Tensor x)
        {
            return _enc->forward(x);
        }

        torch::Tensor quantize(torch::Tensor x)
        {
            torch::Tensor z = _enc->forward(x);
            return _quantizer->forward(z);
        }

        torch::Tensor centers()
        {
            return _quantizer->centers();
        }

        torch::Tensor decode(torch::Tensor x)
        {
            return _dec->forward(x);
        }


    private:
        torch::nn::Sequential _enc, _dec;
        Quantizer _quantizer;
    };

    TORCH_MODULE_TEMPLATED(VQVAE);

    template <typename Quantizer>
    class MLVQVAEImpl : public torch::nn::Module {
    public:
        MLVQVAEImpl(torch::nn::Sequential encoder, torch::nn::Sequential decoder, Quantizer quantizer) : _enc(std::move(encoder)), _dec(std::move(decoder)), _quantizer(std::move(quantizer))
        {
            register_module("Enc", _enc);
            register_module("Quantizer", _quantizer);
            register_module("Dec", _dec);
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x)
        {
            torch::Tensor z = _enc->forward(x);
            torch::Tensor reconstructed = _dec->forward(z);

            torch::Tensor quantized_z, quantization_loss;
            std::tie(quantized_z, quantization_loss) = _quantizer->forward(z);
            torch::Tensor quantized_reconstructed = _dec->forward(quantized_z);
            return {reconstructed, quantized_reconstructed, quantization_loss};
        }

        torch::Tensor encode(torch::Tensor x)
        {
            return _enc->forward(x);
        }

        torch::Tensor quantize(torch::Tensor x)
        {
            torch::Tensor z = _enc->forward(x);
            return _quantizer->forward(z);
        }

        torch::Tensor centers()
        {
            return _quantizer->centers();
        }

        torch::Tensor decode(torch::Tensor x)
        {
            return _dec->forward(x);
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> loss_fn(torch::Tensor x_hat, torch::Tensor x_quantized_hat, torch::Tensor x, torch::Tensor vq_loss)
        {
            torch::Tensor loss, x_rec, x_rec_quantized;
            x_rec = torch::nn::functional::mse_loss(x_hat, x);
            x_rec_quantized = torch::nn::functional::mse_loss(x_quantized_hat, x);
            loss = x_rec + x_rec_quantized + vq_loss;
            return {loss, x_rec, x_rec_quantized};
        }

        void step(std::shared_ptr<torch::optim::Optimizer> optimizer, boost::optional<std::shared_ptr<torch::optim::LRScheduler>> scheduler = boost::none)
        {
            int batch_size = 64;
            float sum_loss = 0.f;
            float sum_quantization_loss = 0.f;
            float sum_loss_ae = 0.f;
            float sum_loss_vq = 0.f;
            // int num_batches = std::ceil(dataset.size()[0] / batch_size);
            int num_batches = 4;
            float denom = num_batches + 1.f;

            // dataset.reset_idx();

            for (int i = 0; i < num_batches; ++i) {
                optimizer->zero_grad();
                // torch::Tensor batch = dataset.sample(batch_size, false);
                torch::Tensor batch = torch::rand({64, 1, 28, 28}).to(torch::Device(torch::kCUDA));
                torch::Tensor x_reconstructed, x_quantized_reconstructed, quantization_loss;
                std::tie(x_reconstructed, x_quantized_reconstructed, quantization_loss) = this->forward(batch);
                torch::Tensor loss, loss_ae, loss_vq;
                std::tie(loss, loss_ae, loss_vq) = loss_fn(x_reconstructed, x_quantized_reconstructed, batch, quantization_loss);
                loss.backward();
                optimizer->step();

                sum_loss += loss.item<float>();
                sum_quantization_loss += torch::Tensor(quantization_loss).item<float>();
                sum_loss_ae += (loss_ae * batch_size).item<float>();
                sum_loss_vq += loss_vq.item<float>();
            }
            // if (_scheduler)
            //     _scheduler.value().step();
            float avg_loss = sum_loss / denom;
            float avg_quantization_loss = sum_quantization_loss / denom;
            float avg_loss_ae = sum_loss_ae / denom;
            float avg_loss_vq = sum_loss_vq / denom;
            std::cout << "Epoch: " << (1) << " Loss: " << avg_loss << " Reconstruction Loss: " << avg_loss_ae << " Reconstruction Quant Loss: " << avg_loss_vq << " Quantization Loss: " << avg_quantization_loss << std::endl;
        }

    private:
        torch::nn::Sequential _enc, _dec;
        Quantizer _quantizer;
    };
    TORCH_MODULE_TEMPLATED(MLVQVAE);
} // namespace nn
