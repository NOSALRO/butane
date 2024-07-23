#pragma once

#include <torch/torch.h>

namespace data {

    struct DataloaderIterator {
        Dataset& _dataset;
        std::vector<torch::Tensor>::iterator _current;

        DataloaderIterator(Dataset& dataset, std::vector<torch::Tensor>::iterator current)
            : _dataset(dataset), _current(current) {}

        DataloaderIterator& operator++()
        {
            ++_current;
            return *this;
        }

        bool operator!=(const DataloaderIterator& other) const { return _current != other._current; }
        Batch operator*() const { return _dataset.get(*_current); }
    };

    class Dataloader {
    public:
        Dataloader(Dataset& dataset, int batch_size, bool shuffle = true)
            : _dataset(dataset), _batch_size(batch_size), _shuffle(shuffle)
        {
            _shuffle_and_split();
        }

        DataloaderIterator begin()
        {
            _shuffle_and_split();
            return {_dataset, _indices.begin()};
        }
        DataloaderIterator end() { return {_dataset, _indices.end()}; }

        size_t batches() { return _indices.size(); }

    private:
        void _shuffle_and_split()
        {
            _indexes = _shuffle ? torch::randperm(_dataset.size()) : torch::arange(_dataset.size());
            _indices = torch::split(_indexes, _batch_size);
        }

        Dataset& _dataset;
        int _batch_size;
        bool _shuffle;
        torch::Tensor _indexes;
        std::vector<torch::Tensor> _indices;
    };
} // namespace data
