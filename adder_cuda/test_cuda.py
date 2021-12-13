import torch
import adder_cuda
import time

def conv_cuda(X_col, W_col):
    # W_col : Co, CiKhKw
    # X_col : CiKhKw, HoWoN
    Co = W_col.size(0)
    HoWoN = X_col.size(1)
    out = torch.zeros((Co, HoWoN), device=torch.device('cuda:0'))

    adder_cuda.ADDER_CONV(X_col, W_col, out)
    
    return out

def conv_no_cuda(X_col, W_col):
    # W_col : Co, CiKhKw
    # X_col : CiKhKw, HoWoN
    out = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
    return out

def conv_weight_cuda(X_col, W_col, grad_output):
    
    return 0

def conv_weight_no_cuda(X_col, W_col, grad_output):
    # W_col : Co, CiKhKw
    # X_col : CiKhKw, HoWoN
    # grad_output : Co, NHoWo
    grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
    return grad_W_col

def test_conv(N,Co,Ci,K,HWi,HWo,stride,padding,CHECK_TRUTH):
    w = torch.randn((Co, Ci* K* K), device=torch.device('cuda:0'))*5
    x = torch.randn((Ci* K* K, HWo * HWo * N), device=torch.device('cuda:0'))*5
    # print(w)
    # print(x)

    start = time.time()
    cuda_result = conv_cuda( x, w)
    end = time.time()
    print("cuda time:", end - start)
    # print(cuda_result)

    if CHECK_TRUTH:
        start = time.time()
        ground_truth = conv_no_cuda(x, w)
        end = time.time()
        print("CPU time:", end - start)
        # print(ground_truth)

        sub = (ground_truth - cuda_result).view(-1)
        print("check result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))

def test_conv_weight(N,Co,Ci,K,HWi,HWo,stride,padding,CHECK_TRUTH):
    grad_y = torch.rand((Co,  HWo * HWo * N), device=torch.device('cuda:0'))
    w = torch.rand((Co, Ci* K* K), device=torch.device('cuda:0'))
    x = torch.rand((Ci* K* K, HWo * HWo * N), device=torch.device('cuda:0'))

    start = time.time()
    cuda_result = conv_weight_cuda(x, w, grad_y)
    end = time.time()
    print("cuda time:", end - start)
    # print(cuda_result)

    if CHECK_TRUTH:
        start = time.time()
        ground_truth = conv_weight_no_cuda(x, w, grad_y)
        end = time.time()
        print("CPU time:", end - start)
        # print(ground_truth)

        sub = (ground_truth - cuda_result).view(-1)
        print("check result:", torch.sum(sub), torch.var(sub), torch.max(sub), torch.min(sub))


if __name__ == "__main__":
    CHECK_TRUTH = 1
    N = 128
    Co = 64
    Ci = 64
    K = 3
    HWi = 8
    HWo = 8
    stride = 1
    padding = 0

    test_conv(N,Co,Ci,K,HWi,HWo,stride,padding,CHECK_TRUTH)

    test_conv_weight(N,Co,Ci,K,HWi,HWo,stride,padding,CHECK_TRUTH)
