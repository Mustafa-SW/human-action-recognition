import torch

print("=== PyTorch Basics ===")

x = torch.tensor([1, 2, 3])

print("Tensor:")
print(x)

matrix = torch.rand(3, 3)

print("\nRandom Matrix:")
print(matrix)

print("\nShape:")
print(matrix.shape)

print("\nMean:")
print(matrix.mean())