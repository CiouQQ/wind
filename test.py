import torch
n = 2
idx = torch.arange(n*n).view(n, n)
obs = torch.randint(0, 11, (8, 2, 4, 4))
edge_index = []
for i in range(n):
    for j in range(n):
        if i < n-1:
            edge_index.append([idx[i, j], idx[i+1, j]])
            edge_index.append([idx[i + 1, j], idx[i, j]])
        if j < n-1:
            edge_index.append([idx[i, j], idx[i, j+1]])
            edge_index.append([idx[i, j + 1], idx[i, j]])
print(torch.tensor(edge_index).t().contiguous().shape)
edge_index = torch.tensor(edge_index).t().contiguous()

# 從 obs 中提取 edge weights
# 轉換 edge_index 的一維索引到二維索引
edge_weights = obs[:, 1, edge_index[0], edge_index[1]]
print(obs[:, 1, :, :].shape)
print(edge_weights.shape)