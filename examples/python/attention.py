import torch
import butane

if __name__ == "__main__":
    dev = torch.device('cuda')

    pemb = butane.nn.PatchEmbeddings([1,100,100], 10, [10, 10])
    emb = pemb(torch.rand(1, 1, 100, 100))

    attention = butane.nn.SelfAttention(10, n_heads = 5)
    print(attention)

    attention_mask = torch.zeros((1, 10, 10))
    print(attention(emb))
