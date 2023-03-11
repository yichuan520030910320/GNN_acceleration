import torch
import time

SIZE = 128
N = SIZE
C = SIZE
H = SIZE
W = SIZE
import random
shape = (N, C, H, W)

# Allocate a tensor on CPU
cpu_tensor = torch.randn(*shape)

# Allocate a tensor on CPU pinned memory
pinned_tensor = torch.empty(*shape, pin_memory=True)
pinned_tensor.copy_(cpu_tensor)



# Define a function to test torch.index_select with and without pinned memory
def test_index_select(use_pin_memory):
    # Allocate a tensor for index selection
    if use_pin_memory:
        idx_tensor = torch.tensor([i % N for i in range(N)], device='cpu', pin_memory=True)
        # idx_tensor = torch.tensor([i % N for i in range(N)], device='cpu')
        
    else:
        idx_tensor = torch.tensor([i % N for i in range(N)], device='cpu')
    # print("Index tensor : {}".format(idx_tensor))
    # Test the time to perform index selection with and without pinned memory
    torch.cuda.synchronize()
    start_time = time.time()
    if use_pin_memory:
        selected_tensor = torch.index_select(pinned_tensor, 0, idx_tensor)
    else:
        selected_tensor = torch.index_select(cpu_tensor, 0, idx_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    print("Time with pinned memory: {:.4f} seconds".format(end_time - start_time) if use_pin_memory else "Time without pinned memory: {:.4f} seconds".format(end_time - start_time))


for i in range(10):
    # Test with pinned memory
    test_index_select(use_pin_memory=True)

    # Test without pinned memory
    test_index_select(use_pin_memory=False)
