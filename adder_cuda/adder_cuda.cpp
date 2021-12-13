/*
All contributions by LY:
Copyright (c) 2020 LY
All rights reserved.
*/
#include <torch/extension.h>
#include <pybind11/pybind11.h>

void ADDER_CONV_GPU(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y);

void ADDER_CONV_WEIGHT_GPU(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_w);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void ADDER_CONV(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(y);
    ADDER_CONV_GPU(x, w, y);
}

void ADDER_CONV_WEIGHT(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_w)
{
    CHECK_INPUT(grad_y);
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_w);
    ADDER_CONV_WEIGHT_GPU(grad_y, x, w, grad_w);
}

PYBIND11_MODULE(adder_cuda, m) {
    m.def("ADDER_CONV", &ADDER_CONV, "ADDER_CONV kernel(CUDA)");
    m.def("ADDER_CONV_WEIGHT", &ADDER_CONV_WEIGHT, "ADDER_CONV_WEIGHT kernel(CUDA)");
}