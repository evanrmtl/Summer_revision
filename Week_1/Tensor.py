import torch
import numpy as np

# creating from a python  array
data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)

# creating from a numpy array
np_data = np.arange(4).reshape((2,2))
x_np = torch.from_numpy(np_data)
print(x_np)

#creatinmg from existing Tensor with new value
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

x_same = torch.full_like(x_data, 3)
print(f"Same Tensor: \n {x_same} \n")

# shape 
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape) 

# Or

rand_wo_shape_tensor = torch.rand(2,3,4)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")
print(f"Random w/o using variable :  \n {rand_wo_shape_tensor}")


# Attribute of a Tensor

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# Operations on Tensors

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
print(f"Tensor device after moving : {tensor.device}")


tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations


# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Tensor to 1 int

agg = tensor.sum() # As to be 1 elem to use item 
agg_item = agg.item()
print(agg_item, type(agg_item)) 

# In place operations

print(tensor, "\n")
tensor.add_(5)
print(tensor)


# Tensor to NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor

