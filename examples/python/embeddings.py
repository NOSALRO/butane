import torch
import butane
import matplotlib.pyplot as plt

if __name__ == "__main__":
    emb = butane.nn.embeddings.SinusoidalEmbeddings(18).to('cuda')
    x = torch.randn(100, 64, 1).to('cuda')
    print(emb(x).size())

    emb = butane.nn.embeddings.LearnableEmbeddings(18).to('cuda')
    x = torch.randn(100, 64, 1).to('cuda')
    print(emb(x).size())
