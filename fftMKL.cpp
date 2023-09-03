#include <iostream>
#include <cstring>
#include <mkl.h>
#include <chrono>
#include "fftw/fftw3.h"

#define THRESHOLD (1e-6)
#define SIZE 2048
#define NRUNS 1000

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

int main() {

	// config
	int size2d = SIZE * SIZE;
	MKL_LONG dimSizes[2] = { SIZE, SIZE };
	std::chrono::duration<float> tFFTW3Plan, tFFTW3, tMKL;  // timers

	// input and output
	float*         in       = new float[size2d];
	fftwf_complex* outFFTW3 = new fftwf_complex[size2d];
	MKL_Complex8*  outMKL   = new MKL_Complex8[size2d];  // single precision

	// Initialize random data
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MT19937, 114514);
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, size2d, in, 0.0, 1.0);
	vslDeleteStream(&stream);

    
	// part I. Single execution
	// FFTW3
	//// plan
	auto tic0 = std::chrono::steady_clock::now();
	auto r2c = fftwf_plan_dft_r2c_2d(SIZE, SIZE, in, outFFTW3, FFTW_ESTIMATE);
	//// compute
	auto tic = std::chrono::steady_clock::now();
	fftwf_execute(r2c);  // fftwl for double precision
	auto toc = std::chrono::steady_clock::now();
	tFFTW3Plan = tic - tic0;
	tFFTW3 = toc - tic;

	// MKL
	//// config
	DFTI_DESCRIPTOR_HANDLE desc = NULL;
	DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_REAL, 2, dimSizes);
	DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	DftiCommitDescriptor(desc);
	//// compute
	tic = std::chrono::steady_clock::now();
	DftiComputeForward(desc, in, outMKL);
	toc = std::chrono::steady_clock::now();
	tMKL = toc - tic;

	// compare
	std::cout << "Single execution time (s):" << std::endl;
	std::cout << "FFTW3: " << tFFTW3.count() << "\t(+Plan: " << tFFTW3Plan.count() << ")" << std::endl;
	std::cout << "MKL:   " << tMKL.count()   << std::endl;
	std::cout << std::endl;


	// Part II. NRUNS(1000) executions
	std::cout << "Average time over " << NRUNS << " executions (s):" << std::endl;

	tic = std::chrono::steady_clock::now();
	for (int i = 0; i < NRUNS; i++) {
		fftwf_execute(r2c);
	}
	toc = std::chrono::steady_clock::now();
	tFFTW3 = toc - tic;
	std::cout << "FFTW3: " << tFFTW3.count() / NRUNS << std::endl;

	tic = std::chrono::steady_clock::now();
	for (int i = 0; i < NRUNS; i++) {
		DftiComputeForward(desc, in, outMKL);
	}
	toc = std::chrono::steady_clock::now();
	tMKL = toc - tic;
	std::cout << "MKL:   " << tMKL.count() / NRUNS << std::endl;
	std::cout << std::endl;

	
	// Part III. Compare results
	bool resultsMatch = compare(outFFTW3, outMKL, dimSizes);
	std::cout << "Results are " << (resultsMatch?"correct":"incorrect") << std::endl;
	
	return 0;
}

