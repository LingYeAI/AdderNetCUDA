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
    Co = W_col.size(0)
    CiKhKw = W_col.size(1)
    HoWoN = X_col.size(1)
    out = torch.zeros((Co, CiKhKw), device=torch.device('cuda:0'))

    adder_cuda.ADDER_CONV_WEIGHT(grad_output, X_col, W_col, out)
    return out


def conv_weight_no_cuda(X_col, W_col, grad_output):
    # W_col : Co, CiKhKw
    # X_col : CiKhKw, HoWoN
    # grad_output : Co, HoWoN
    grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))
                  * grad_output.unsqueeze(1)).sum(2)
    return grad_W_col


def conv_input_cuda(X_col, W_col, grad_output):
    out = torch.zeros_like(X_col)
    adder_cuda.ADDER_CONV_INPUT(grad_output, X_col, W_col, out)
    return out


def conv_backward_cuda(X_col, W_col, grad_output):
    grad_w = torch.zeros_like(W_col)
    grad_i = torch.zeros_like(X_col)
    adder_cuda.ADDER_BACKWARD(grad_output, X_col, W_col, grad_w, grad_i)
    return grad_w, grad_i


def conv_input_no_cuda(X_col, W_col, grad_output):
    grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)
                    ).clamp(-1, 1)*grad_output.unsqueeze(1)).sum(0)
    return grad_X_col


def test_conv(N, Co, Ci, K, HWi, HWo, stride, padding, CHECK_TRUTH):
    w = torch.randn((Co, Ci * K * K), device=torch.device('cuda:0'))*5
    x = torch.randn((Ci * K * K, HWo * HWo * N),
                    device=torch.device('cuda:0'))*5
    # print(w)
    # print(x)

    start = time.time()
    cuda_result = conv_cuda(x, w)
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
        print("check result:", torch.sum(sub), torch.var(
            sub), torch.max(sub), torch.min(sub))


def test_conv_weight(N, Co, Ci, K, HWi, HWo, stride, padding, CHECK_TRUTH):
    grad_y = torch.randn((Co,  HWo * HWo * N), device=torch.device('cuda:0'))
    w = torch.randn((Co, Ci * K * K), device=torch.device('cuda:0'))
    x = torch.randn((Ci * K * K, HWo * HWo * N), device=torch.device('cuda:0'))

    start = time.time()
    cuda_result = conv_weight_cuda(x, w, grad_y)
    end = time.time()
    print("cuda time:", end - start)
    print(cuda_result)

    if CHECK_TRUTH:
        start = time.time()
        ground_truth = conv_weight_no_cuda(x, w, grad_y)
        end = time.time()
        print("CPU time:", end - start)
        print(ground_truth)

        sub = (ground_truth - cuda_result).view(-1)
        print("check result:", torch.sum(sub), torch.var(
            sub), torch.max(sub), torch.min(sub))


def test_conv_input(N, Co, Ci, K, HWi, HWo, stride, padding, CHECK_TRUTH):
    grad_y = torch.randn((Co,  HWo * HWo * N), device=torch.device('cuda:0'))
    w = torch.randn((Co, Ci * K * K), device=torch.device('cuda:0'))
    x = torch.randn((Ci * K * K, HWo * HWo * N), device=torch.device('cuda:0'))

    start = time.time()
    cuda_result = conv_input_cuda(x, w, grad_y)
    end = time.time()
    print("cuda time:", end - start)
    print(cuda_result)

    if CHECK_TRUTH:
        start = time.time()
        ground_truth = conv_input_no_cuda(x, w, grad_y)
        end = time.time()
        print("CPU time:", end - start)
        print(ground_truth)

        sub = (ground_truth - cuda_result).view(-1)
        print("check result:", torch.sum(sub), torch.var(
            sub), torch.max(sub), torch.min(sub))


def test_backward(N, Co, Ci, K, HWi, HWo, stride, padding, CHECK_TRUTH):
    grad_y = torch.randn((Co,  HWo * HWo * N), device=torch.device('cuda:0'))
    w = torch.randn((Co, Ci * K * K), device=torch.device('cuda:0'))
    x = torch.randn((Ci * K * K, HWo * HWo * N), device=torch.device('cuda:0'))

    start = time.time()
    cuda_grad_w, cuda_grad_i = conv_backward_cuda(x, w, grad_y)
    end = time.time()
    print("cuda time:", end - start)
    # print(cuda_result)

    if CHECK_TRUTH:
        start = time.time()
        gt_grad_w = conv_weight_no_cuda(x, w, grad_y)
        gt_grad_i = conv_input_no_cuda(x, w, grad_y)
        end = time.time()
        print("CPU time:", end - start)
        # print(ground_truth)

        sub_w = (cuda_grad_w - gt_grad_w).view(-1)
        sub_i = (cuda_grad_i - gt_grad_i).view(-1)
        print("check grad_w result:", torch.sum(sub_w), torch.var(
            sub_w), torch.max(sub_w), torch.min(sub_w))
        print("check grad_i result:", torch.sum(sub_i), torch.var(
            sub_i), torch.max(sub_i), torch.min(sub_i))


if __name__ == "__main__":

    CHECK_TRUTH = 1

    N = 16
    Co = 32
    Ci = 32
    K = 3
    HWi = 8
    HWo = 8
    stride = 1
    padding = 0

    test_conv(N,Co,Ci,K,HWi,HWo,stride,padding,CHECK_TRUTH)

    test_backward(N, Co, Ci, K, HWi, HWo, stride, padding, CHECK_TRUTH)
