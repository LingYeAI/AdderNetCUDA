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
                // if (threadIdx.x == 0 && threadIdx.y == 0){
                //     for (int i = 0; i < BLOCK_SIZE; i++){
                //         for (int j =0; j < BLOCK_SIZE; j++){
                //             if (i + cikhkw_ < CiKhKw && j + cikhkw_ < CiKhKw){
                //                 SW[i][j] = W[CiKhKw * (co_ + j) + cikhkw_ + i];
                //                 SX[i][j] = X[(cikhkw_ + j) * NHoWo + nhowo_ + i];
                //             }
                //             else{
                //                 SW[threadIdx.y][threadIdx.x] = 0;
                //                 SX[threadIdx.y][threadIdx.x] = 0;
                //             }
                //         }
                //     }
                // }

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