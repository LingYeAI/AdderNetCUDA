/*
All contributions by LY:
Copyright (c) 2020 LY
All rights reserved.
*/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <vector>
#define BLOCK_SIZE 16
#define MAX_THREADS 1024
#define MAX_BLOCKS 65535
#define MAX_KW 3
#define MAX_KH 3
#define MULTIPLE 3

__global__ void CONV(
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ output,
    const int CiKhKw,
    const int NHoWo,
    const int Co)
{
    /*
    *   输入：W: Co, CiKhKw
             X: CiKhKw, HoWoN
    *   输出： output: Co, HoWoN 
    *   block/thread: (Co/BLOCK_SIZE, min(NHoWo/BLOCK_SIZE, MAX_BLOCKS))/(BLOCK_SIZE, BLOCK_SIZE)
    *   计算: -|w - x|
    */
    for(int nhowo_ = blockIdx.y * blockDim.y + threadIdx.y; nhowo_ < NHoWo; nhowo_ += gridDim.y * blockDim.y){
        for(int co_ = blockIdx.x * blockDim.x + threadIdx.x; co_ < Co; co_ += gridDim.x * blockDim.x){
            __shared__ float SW[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float SX[BLOCK_SIZE][BLOCK_SIZE];

            float* Cfinal = &output[co_ * NHoWo + nhowo_];
            float Cvalue = 0.0;
            float c = 0.0;
            for (int cikhkw_=0; cikhkw_<CiKhKw; cikhkw_+=BLOCK_SIZE) {

                if (cikhkw_ + threadIdx.y < CiKhKw && cikhkw_ + threadIdx.x < CiKhKw){
                    SW[threadIdx.y][threadIdx.x] = W[co_ * CiKhKw + cikhkw_ + threadIdx.y];
                    SX[threadIdx.y][threadIdx.x] = X[(cikhkw_ + threadIdx.x) * NHoWo + nhowo_];
                }
                else{
                    SW[threadIdx.y][threadIdx.x] = 0;
                    SX[threadIdx.y][threadIdx.x] = 0;
                }               

                __syncthreads();

                // printf("%d %d %f %f\n",threadIdx.x,threadIdx.y,SW[threadIdx.x][threadIdx.y],SX[threadIdx.x][threadIdx.y]);
                for (int inner_cikhkw=0; inner_cikhkw<BLOCK_SIZE; inner_cikhkw++){
                    float w_x = SW[inner_cikhkw][threadIdx.x] - SX[threadIdx.y][inner_cikhkw];
                    w_x = (w_x < 0) ? w_x : -w_x;
                    float psum = w_x - c;
                    float t = psum + Cvalue;
                    c = t - Cvalue - psum;
                    Cvalue = t;
                    
                }
                __syncthreads();
            }
            *Cfinal = Cvalue;
        }
    }
}

__global__ void CONV_WEIGHT(
    const float* __restrict__ grad_y,
    const float* __restrict__ X,
    const float* __restrict__ W,
    float* __restrict__ grad_w,
    const int CiKhKw,
    const int NHoWo,
    const int Co)
{
    /*
    *   输入：W: Co, CiKhKw
             X: CiKhKw, HoWoN
             grad_y: Co, HoWoN
    *   输出： grad_w: Co, CiKhKw
    *   grid/block/thread: 1/(Co, NHoWo)/(BLOCK_SIZE, BLOCK_SIZE)
    *   计算: grad_y * (x - w)
    */
    int thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_num_y = gridDim.y * blockDim.y;
    int thread_num_x = gridDim.x * blockDim.x;

    for(int nhowo_ = thread_y; nhowo_ < NHoWo; nhowo_ += thread_num_y){
        for(int co_ = thread_x; co_ < Co; co_ += thread_num_x){
            __shared__ float SW[BLOCK_SIZE][BLOCK_SIZE]; //shared weight
            __shared__ float SX[BLOCK_SIZE][BLOCK_SIZE]; //shared input
            __shared__ float SG[BLOCK_SIZE][BLOCK_SIZE]; //shared grad_y
            __shared__ float SP[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE]; //shared psum

            // float* Cfinal = &grad_w[co_ * CiKhKw + nhowo_];
            // float Cvalue = 0.0;
            // float c = 0.0;

            SG[threadIdx.y][threadIdx.x] = grad_y[co_  * NHoWo + nhowo_];

            for (int cikhkw_=0; cikhkw_<CiKhKw; cikhkw_+=BLOCK_SIZE) {

                float* Cfinal = &grad_w[co_ * CiKhKw + threadIdx.y];

                if (cikhkw_ + threadIdx.y < CiKhKw && cikhkw_ + threadIdx.x < CiKhKw){
                    SW[threadIdx.y][threadIdx.x] = W[co_ * CiKhKw + cikhkw_ + threadIdx.y];
                    SX[threadIdx.y][threadIdx.x] = X[(cikhkw_ + threadIdx.x) * NHoWo + nhowo_];
                }
                else{
                    SW[threadIdx.y][threadIdx.x] = 0;
                    SX[threadIdx.y][threadIdx.x] = 0;
                }               

                __syncthreads();
                
                // printf("%d %d %f %f\n",threadIdx.x,threadIdx.y,SW[threadIdx.x][threadIdx.y],SX[threadIdx.x][threadIdx.y]);
                //y -> NHoWo
                //x -> Co
                for (int inner_cikhkw = 0; inner_cikhkw < BLOCK_SIZE; inner_cikhkw++){
                    SP[threadIdx.y][threadIdx.x][inner_cikhkw] = (SW[inner_cikhkw][threadIdx.x] - SX[threadIdx.y][inner_cikhkw]) * SG[threadIdx.y][threadIdx.x];
                }
                __syncthreads();

                float result = 0.0;
                for (int inner_nhowo = 0; inner_nhowo < BLOCK_SIZE; inner_nhowo++){
                    result += SP[inner_nhowo][threadIdx.x][threadIdx.y];
                    SP[inner_nhowo][threadIdx.x][threadIdx.y] = 0;
                }
                *Cfinal = result;
            }
            
        }
    }
}

void ADDER_CONV_GPU(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor y)
{/*
    *   输入： x: 
    *         w: Co * Ci * Kh * Kw
    *   输出： y: N * Ho * Wo * Co
    *   计算: -|w - x|
    */
    int Co = w.size(0);
    int NHoWo = x.size(1);
    int CiKhKw = w.size(1);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int a1 = Co / BLOCK_SIZE + 1;
    if (a1 > MAX_BLOCKS) {
        a1 = MAX_BLOCKS;   
    }
    int a2 = NHoWo  / BLOCK_SIZE + 1;
    if (a2 > MAX_BLOCKS) {
        a2 = MAX_BLOCKS;
    }
    dim3 gridDim(a1, a2);

    AT_DISPATCH_ALL_TYPES(x.type(), "adder CONV kernel", ([&] {  
        CONV<<<gridDim, blockDim>>>(
        x.data<float>(),
        w.data<float>(),
        y.data<float>(),
        CiKhKw,
        NHoWo,
        Co);
    }));

}

void ADDER_CONV_WEIGHT_GPU(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor grad_w)
{
    /*
    *   输入：W: Co, CiKhKw
             X: CiKhKw, HoWoN
             grad_y: Co, HoWoN
    *   输出： grad_w: Co, CiKhKw
    *   grid/block/thread: 1/(Co, NHoWo)/(BLOCK_SIZE, BLOCK_SIZE)
    *   计算: grad_y * (x - w)
    */

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int Co = w.size(0);
    int CiKhKw = w.size(1);
    int NHoWo = x.size(1);

    dim3 gridDim(Co, NHoWo);  
    AT_DISPATCH_ALL_TYPES(x.type(), "conv weight kernel", ([&] {
        CONV_WEIGHT<<<gridDim, blockDim>>>(
            x.data<float>(),
            grad_y.data<float>(),
            w.data<float>(),
            grad_w.data<float>(),
            CiKhKw,
            NHoWo,
            Co
        );
      }));
}