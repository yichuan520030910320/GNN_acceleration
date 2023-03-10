import torch
import time
avg_wo_pin=[]
avg_w_pin=[]

SIZE=128
N=SIZE
C=SIZE
H=SIZE
W=SIZE
import random
shape = (N, C, H, W)

# Allocate a tensor on CPU
cpu_tensor = torch.randn(*shape)

# Allocate a tensor on CPU pinned memory
pinned_tensor = torch.empty(*shape, pin_memory=True)
pinned_tensor.copy_(cpu_tensor)

# Allocate a tensor on GPU
device = torch.device('cuda')
gpu_tensor = torch.empty(*shape).to(device)

torch.cuda.synchronize()
for i in range(10):
    cpu_tensor += random.random() - 0.5
    pinned_tensor += random.random() - 0.5

    # Test the time to transfer the tensor from CPU to GPU without pinned memory
    torch.cuda.synchronize()
    start_time = time.time()
    gpu_tensor.copy_(cpu_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    print("Transfer time without pinned memory: {:.4f} seconds".format(end_time - start_time))
    avg_wo_pin.append(end_time - start_time)

    # Test the time to transfer the tensor from CPU to GPU with pinned memory
    torch.cuda.synchronize()
    start_time = time.time()
    gpu_tensor.copy_(pinned_tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    print("Transfer time with pinned memory: {:.4f} seconds".format(end_time - start_time))
    avg_w_pin.append(end_time - start_time)
avg_w_pin=sum(avg_w_pin)/len(avg_w_pin)
avg_wo_pin=sum(avg_wo_pin)/len(avg_wo_pin)
print("Average time without pinned memory: {:.4f} mseconds".format(avg_wo_pin*1000))
print("Average time with pinned memory: {:.4f} mseconds".format(avg_w_pin*1000))

print('ratio of time with pinned memory to without pinned memory: {:.4f} '.format(avg_w_pin/avg_wo_pin))
#128 128 128 128
# Average time without pinned memory: 0.8317 seconds
# Average time with pinned memory: 0.0902 seconds