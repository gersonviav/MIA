import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_arch_list())  # debe aparecer sm_120
import torch
x = torch.randn(8192, 8192, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(y.mean().item())