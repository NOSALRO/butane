#pragma once

#include "utils.hpp"

class AEImpl : public torch::nn::Module {
public:
    AEImpl(torch::nn::Sequential encoder, torch::nn::Sequential decoder, int latent_dim) : _enc(std::move(encoder)), _dec(std::move(decoder)), _latent_dim(latent_dim)
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
    int _latent_dim;
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

private:
    torch::nn::Sequential _enc, _dec;
    Quantizer _quantizer;
};

TORCH_MODULE_TEMPLATED(MLVQVAE);
