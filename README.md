## Training addernet accelerated by CUDA

### Usage
```
cd adder_cuda
python setup.py install
cd ..
python main.py
```

### Environment
pytorch 1.10.0
CUDA 11.3

### benchmark

| version                                                   | training_time_per_batch/s |
| --------------------------------------------------------- | ------------------------- |
| [raw](https://github.com/huawei-noah/AdderNet)            | 1.61                      |
| torch.cdist                                               | 1.49                      |
| [cuda_unoptimized](https://github.com/jdnie/AdderNetCuda) | 0.4508                    |
| this work                                                 | 0.3158                    |

The CUDA version of AdderNet has achieved a 5× speed increase over the original version. There seems to be some bugs in the Cuda_unoptimized version, causing the model to fail to converge. Its speed is still listed here for comparison. The experiment was run on RTX 2080Ti platform, and ResNet-20 based on CIFAR-10 was trained.


|Time(%)|	Time	|Calls	|Avg	    |Min	    |Max	    |Name|
|-------|-----------|-------|-----------|-----------|-----------|----|
|48.57	|30.4752s	|3920	|7.7743ms	|162.70us	|12.271ms	|CONV_BACKWARD|
|4.85	|21.8686s	|19680	|1.1112ms	|5.3770us	|11.827ms	|_ZN2at6native27unrolled_elementwise_kernel...|
|7.46	|4.67901s	|5920	|790.37us	|26.529us	|1.5841ms	|CONV|
|2.24	|1.40372s	|3920	|358.09us	|31.298us	|845.80us	|col2im_kernel|
|2.10	|1.31882s	|36862	|35.777us	|1.4720us	|276.24us	|vectorized_elementwise_kernel|
|1.43	|900.03ms	|5920	|152.03us	|7.9040us	|372.40us	|im2col_kernel|

Here is the time distribution of training an epoch. If you are interested, you can continue to optimize the CUDA kernel.