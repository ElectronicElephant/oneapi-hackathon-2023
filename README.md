Intel oneAPI Hackathon 2023
===

## Hackathon 问题描述

> 摘自 BGP-Hackathon-Intro

使用 oneMKL 工具，对 FFT 算法进行加速与优化。
1. 下载和使用最新版本 oneMKL
2. 调用 oneMKL 相应 API 函数， 产生 2048 * 2048 个 随机单精度实数；
3. 根据 2 产生的随机数据作为输入，实现两维 Real to complex FFT 参考代码；
4. 根据 2 产生的随机数据作为输入， 调用 oneMKL API 计算两维 Real to complex FFT；
5. 结果正确性验证，对 3 和 4 计算的两维 FFT 输出数据进行全数据比对（允许适当精度误差）， 输出 “结果正确”或“结果不正确”信息；
6. 平均性能数据比对（比如运行 1000 次），输出 FFT 参考代码平均运行时间和 oneMKL FFT 平均运行时间。

## 环境配置

该 Hackathon 需要两个组件：

- Intel® oneAPI Base Toolkit
    - [链接](https://www.intel.cn/content/www/cn/zh/developer/tools/oneapi/base-toolkit-download.html)
    - 提供 ICC 编译器和其他基础工具
- Intel® oneAPI Math Kernel Library
    - [链接](https://www.intel.cn/content/www/cn/zh/developer/tools/oneapi/onemkl-download.html)
    - 提供了计算 FFT 所需要的 MKL 库

笔者选择了 `Offline Installer for Linux 2023.2.0`。在无头服务器上，可以使用以下命令执行静默安装：
```
./l_BaseKit_p_2023.2.0.49397_offline.sh \
    -a \
    -s \
    --install-dir /mnt/public/lib/oneapi \
    --eula accept
```
，安装完成后，可使用以下命令激活 oneAPI 环境：
```
. /mnt/public/lib/oneapi/setvars.sh
```

## 初认 oneAPI 之 Hello ICC

编辑 `test.cpp` 文件
```
#include <iostream>
using namespace std;
 
int main() 
{
    cout << "Hello, ICC!";
    return 0;
}
```

编译
```
icpx -fsycl -I${MKLROOT}/include -c test.cpp -o test.o
```

链接
```
icpx -fsycl test.o -fsycl-device-code-split=per_kernel \
        "${MKLROOT}/lib/intel64"/libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group \
        "${MKLROOT}/lib/intel64"/libmkl_intel_ilp64.a \
        "${MKLROOT}/lib/intel64"/libmkl_sequential.a \
        "${MKLROOT}/lib/intel64"/libmkl_core.a -Wl,--end-group -lsycl -lOpenCL \
        -lpthread -lm -ldl -o test.out
```

## 一些关键点

### 计时

`C++ 11` 为我们提供了 `chrono` 高精度时钟，实例用法如下:

```
tic = std::chrono::steady_clock::now();
// Do something
toc = std::chrono::steady_clock::now();
delta_t = toc - tic
std::cout << delta_t.count() << std::endl;  // in second
```

### 原地 FFT 计算的 CCE 格式

我们注意到实数到虚数的 FFT 计算产生的结果是共轭的。出于类似的原因， oneAPI 原地 FFT 计算默认会使用 CCE 格式来存储结果。通俗来说，只保存了“一半”的结果。下面提供一个实例函数，可以将 oneAPI 产生的结果与 fftw3 的结果相比较。

```
#define THRESHOLD (1e-6)

bool compare(fftwf_complex* outFFTW3, MKL_Complex8* outMKL, MKL_LONG dimSizes[2]) {
	auto height = dimSizes[0];
	auto width = dimSizes[1];

	for (int i=0; i<height; i++) {
		for (int j=0; j< (width/2+1); j++) {
			auto idxFFTW  = i*(width/2+1) + j;
			auto idxMKL   = i*   width    + j;
			auto diff = std::abs(outFFTW3[idxFFTW][0] - outMKL[idxMKL].real)
					  + std::abs(outFFTW3[idxFFTW][1] - outMKL[idxMKL].imag);
			if (diff > THRESHOLD) {
				std::cout << "Error: " << diff << " at " << i << ", " << j << std::endl;
				return false;
			}
		}
	}
	return true;
}
```

### oneAPI 中提供的 FFTW3 性能问题

出于兼容性与易用性考虑，oneAPI 中同样提供了 FFTW3 兼容接口。需要注意的是，该接口背后的计算过程依然是由 MKL 加速的，并非是开源 FFTW3 实现。通过实验也可以发现，通过 oneAPI 的 FFTW3 接口计算所得结果可以与直接使用 oneAPI 对齐，计算所需时间也相近。
